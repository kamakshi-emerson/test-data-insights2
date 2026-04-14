try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
    from observability.config import settings as _obs_settings
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]
    class _ObsSettingsStub:
        AGENT_NAME: str = 'Data Insights Assistant for Non-Technical Users'
        PROJECT_NAME: str = 'Data Insights Project'
    _obs_settings = _ObsSettingsStub()

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, constr
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Observability wrappers are injected by the runtime
# from observability import trace_step, trace_step_sync

# Load environment variables from .env if present
load_dotenv()

# ---------------------- Logging Configuration ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("data_insights_agent")

# ---------------------- Configuration Management ----------------------

class Config:
    """Configuration loader for environment variables."""
    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    @staticmethod
    def validate_rag():
        missing = []
        required = [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ]
        for k in required:
            if not os.getenv(k):
                missing.append(k)
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    @staticmethod
    def get_openai_client() -> openai.AsyncOpenAI:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")
        return openai.AsyncOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )

    @staticmethod
    def get_search_client() -> SearchClient:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        if not endpoint or not index_name or not api_key:
            raise RuntimeError("AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, and AZURE_SEARCH_API_KEY must be set")
        return SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )

    @staticmethod
    def get_embedding_deployment() -> str:
        return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    @staticmethod
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_model() -> str:
        return "gpt-4.1"

    @staticmethod
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_temperature() -> float:
        return 0.7

    @staticmethod
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_max_tokens() -> int:
        return 2000

    @staticmethod
    def get_rag_top_k() -> int:
        return 5

    @staticmethod
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_system_prompt() -> str:
        return (
            "You are a Data Insights Assistant for non-technical users. Your role is to answer user questions about data by providing clear, concise, and accurate insights based strictly on the retrieved knowledge base content. Do not use technical jargon or assume prior data knowledge. If the answer cannot be found in the provided documents, politely inform the user and suggest rephrasing their question. Always ensure your responses are easy to understand and directly address the user's query."
        )

    @staticmethod
    def get_fallback_response() -> str:
        return (
            "I'm sorry, I couldn't find an answer to your question in the available data. Please try rephrasing your question or ask about a different topic."
        )

# ---------------------- Pydantic Models ----------------------

class QuestionRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=1, max_length=50000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Question exceeds 50,000 character limit.")
        return v

class AnswerResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    tips: Optional[str] = None

# ---------------------- Logger Utility ----------------------

class Logger:
    """Audit logging for compliance and monitoring."""

    @staticmethod
    def log_event(event: str, level: str = "info", details: Optional[Dict[str, Any]] = None) -> None:
        try:
            msg = f"{event} | Details: {details or {}}"
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "debug":
                logger.debug(msg)
            else:
                logger.info(msg)
        except Exception as e:
            # Logging must not interrupt main flow
            logger.error(f"Logging failed: {e}")

# ---------------------- Error Handler ----------------------

class ErrorHandler:
    """Centralized error handling, fallback logic, and escalation."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def handle_error(self, error_code: str, context: Dict[str, Any]) -> str:
        self.logger.log_event(
            event=f"Error occurred: {error_code}",
            level="error",
            details=context
        )
        if error_code == "NO_DATA_FOUND":
            return Config.get_fallback_response()
        elif error_code == "RETRIEVAL_ERROR":
            return "There was a problem retrieving information. Please try again later."
        elif error_code == "REWRITE_ERROR":
            return "I had trouble simplifying the answer. Here is the original answer."
        else:
            return "An unexpected error occurred. Please try again later."

# ---------------------- Retrieval Client ----------------------

class RetrievalClient:
    """Queries Azure AI Search using semantic/vector search."""

    def __init__(self):
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, top_k: int = 5) -> List[str]:
        async with trace_step(
            "retrieve_chunks", step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r) if r else 0}"
        ) as step:
            try:
                Config.validate_rag()
                search_client = Config.get_search_client()
                openai_client = Config.get_openai_client()
                embedding_resp = await openai_client.embeddings.create(
                    input=query,
                    model=Config.get_embedding_deployment()
                )
                vector_query = VectorizedQuery(
                    vector=embedding_resp.data[0].embedding,
                    k_nearest_neighbors=top_k,
                    fields="vector"
                )
                results = search_client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=top_k,
                    select=["chunk", "title"]
                )
                context_chunks = [r["chunk"] for r in results if r.get("chunk")]
                step.capture(context_chunks)
                return context_chunks
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                step.capture([])
                return []

# ---------------------- LLM Client ----------------------

class LLMClient:
    """Handles prompt construction and calls to OpenAI GPT-4.1."""

    def __init__(self):
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_answer(self, prompt: str, context: str) -> str:
        async with trace_step(
            "generate_answer", step_type="llm_call",
            decision_summary="Call LLM to generate answer",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            openai_client = Config.get_openai_client()
            system_prompt = Config.get_system_prompt()
            user_message = f"{prompt}\n\nRelevant information:\n{context}"
            try:
                _t0 = time.time()
                response = await openai_client.chat.completions.create(
                    model=Config.get_llm_model(),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=Config.get_llm_temperature(),
                    max_tokens=Config.get_llm_max_tokens()
                )
                content = response.choices[0].message.content.strip()
                step.capture(content)
                try:
                    trace_model_call(
                        provider="openai",
                        model_name=Config.get_llm_model(),
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        latency_ms=int((time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                return content
            except Exception as e:
                logger.error(f"LLM call error: {e}")
                step.capture("")
                return ""

# ---------------------- Domain Logic ----------------------

class DomainLogic:
    """Business rules: grounding, simplification, validation."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def validate_grounding(self, answer: str, chunks: List[str]) -> bool:
        async with trace_step(
            "validate_grounding", step_type="process",
            decision_summary="Check if answer is grounded in retrieved content",
            output_fn=lambda r: f"grounded={r}"
        ) as step:
            # Simple heuristic: check if any chunk content is present in answer (case-insensitive)
            answer_lower = answer.lower()
            for chunk in chunks:
                if chunk and chunk.strip() and chunk.lower() in answer_lower:
                    step.capture(True)
                    return True
            # If answer is short, or doesn't overlap, consider not grounded
            step.capture(False)
            return False

    async def simplify_language(self, answer: str) -> str:
        async with trace_step(
            "simplify_language", step_type="process",
            decision_summary="Simplify answer for non-technical users",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            # Use LLM to rewrite answer in plain language
            prompt = (
                "Rewrite the following answer so it is clear, simple, and easy to understand for someone with no technical background. Avoid jargon and explain any complex terms.\n\n"
                f"Answer: {answer}"
            )
            try:
                simplified = await self.llm_client.generate_answer(prompt, "")
                if simplified and len(simplified) > 0:
                    step.capture(simplified)
                    return simplified
                else:
                    step.capture(answer)
                    return answer
            except Exception as e:
                logger.warning(f"Simplification failed: {e}")
                step.capture(answer)
                return answer

# ---------------------- Application Orchestrator ----------------------

class ApplicationOrchestrator:
    """Coordinates the workflow: input validation, retrieval, LLM invocation, error handling, and output formatting."""

    def __init__(self, retrieval_client: RetrievalClient, llm_client: LLMClient, domain_logic: DomainLogic, error_handler: ErrorHandler, logger: Logger):
        self.retrieval_client = retrieval_client
        self.llm_client = llm_client
        self.domain_logic = domain_logic
        self.error_handler = error_handler
        self.logger = logger

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_question(self, question: str) -> str:
        async with trace_step(
            "process_question", step_type="final",
            decision_summary="Main entry point for handling user questions",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                # Input validation and sanitization
                sanitized_question = question.strip()
                if not sanitized_question:
                    step.capture(Config.get_fallback_response())
                    return Config.get_fallback_response()

                # Retrieval
                chunks = await self.retrieval_client.retrieve_chunks(sanitized_question, Config.get_rag_top_k())
                self.logger.log_event(
                    event="Chunks retrieved",
                    level="info",
                    details={"num_chunks": len(chunks)}
                )

                if not chunks:
                    answer = self.error_handler.handle_error("NO_DATA_FOUND", {"question": sanitized_question})
                    step.capture(answer)
                    return answer

                # Summarize and simplify retrieved content for context
                context = "\n\n".join(chunks)
                prompt = f"User question: {sanitized_question}"

                # LLM answer generation
                answer = await self.llm_client.generate_answer(prompt, context)
                if not answer or answer.strip() == "":
                    answer = self.error_handler.handle_error("NO_DATA_FOUND", {"question": sanitized_question})
                    step.capture(answer)
                    return answer

                # Validate grounding
                grounded = await self.domain_logic.validate_grounding(answer, chunks)
                if not grounded:
                    answer = self.error_handler.handle_error("NO_DATA_FOUND", {"question": sanitized_question})
                    step.capture(answer)
                    return answer

                # Simplify language
                simplified = await self.domain_logic.simplify_language(answer)
                if not simplified or simplified.strip() == "":
                    simplified = answer  # fallback to original

                # Output formatting
                step.capture(simplified)
                return simplified
            except Exception as e:
                self.logger.log_event(
                    event="Processing error",
                    level="error",
                    details={"error": str(e)}
                )
                answer = self.error_handler.handle_error("RETRIEVAL_ERROR", {"error": str(e)})
                step.capture(answer)
                return answer

# ---------------------- User Interface Handler ----------------------

class UserInterfaceHandler:
    """Handles incoming user questions and outgoing responses."""

    def __init__(self, orchestrator: ApplicationOrchestrator):
        self.orchestrator = orchestrator

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def submit_question(self, question: str) -> str:
        async with trace_step(
            "submit_question", step_type="parse",
            decision_summary="Receive and sanitize user question",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            sanitized = question.strip()
            step.capture(sanitized)
            return await self.orchestrator.process_question(sanitized)

    async def receive_response(self, response: str) -> str:
        # For future extensibility (e.g., formatting, streaming)
        return response

# ---------------------- Main Agent Class ----------------------

class DataInsightsAgent:
    """Main agent class composing all components."""

    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)
        self.llm_client = LLMClient()
        self.retrieval_client = RetrievalClient()
        self.domain_logic = DomainLogic(self.llm_client)
        self.orchestrator = ApplicationOrchestrator(
            self.retrieval_client,
            self.llm_client,
            self.domain_logic,
            self.error_handler,
            self.logger
        )
        self.ui_handler = UserInterfaceHandler(self.orchestrator)

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_question(self, question: str) -> str:
        return await self.ui_handler.submit_question(question)

# ---------------------- FastAPI App ----------------------

app = FastAPI(
    title="Data Insights Assistant for Non-Technical Users",
    description="API for Data Insights Assistant using Azure AI Search and OpenAI GPT-4.1",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DataInsightsAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "validation_error",
            "error_message": str(exc),
            "tips": "Check your input for missing fields, excessive length, or invalid characters."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "http_error",
            "error_message": exc.detail,
            "tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "internal_error",
            "error_message": "An unexpected error occurred.",
            "tips": "Please try again later or contact support."
        }
    )

@app.post("/ask", response_model=AnswerResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def ask_question(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error_type": "malformed_json",
                "error_message": "Malformed JSON in request body.",
                "tips": "Ensure your JSON is properly formatted with double quotes and valid syntax."
            }
        )
    try:
        req = QuestionRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Input validation failed: {ve}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error_type": "validation_error",
                "error_message": str(ve),
                "tips": "Check your input for missing fields, excessive length, or invalid characters."
            }
        )
    try:
        answer = await agent.answer_question(req.question)
        return AnswerResponse(success=True, answer=answer)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return AnswerResponse(
            success=False,
            error_type="processing_error",
            error_message=str(e),
            tips="Please try again later or contact support."
        )

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# ---------------------- Main Entry Point ----------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Data Insights Assistant API...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)