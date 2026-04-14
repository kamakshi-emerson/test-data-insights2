
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Configuration management for Data Insights Assistant.
    Handles environment variable loading, API key management,
    LLM configuration, domain-specific settings, validation, and fallbacks.
    """

    # Required environment variables for RAG and LLM
    REQUIRED_ENV_VARS = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # LLM configuration defaults
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are a Data Insights Assistant for non-technical users. Your role is to answer user questions about data by providing clear, concise, and accurate insights based strictly on the retrieved knowledge base content. Do not use technical jargon or assume prior data knowledge. If the answer cannot be found in the provided documents, politely inform the user and suggest rephrasing their question. Always ensure your responses are easy to understand and directly address the user's query."
        ),
        "user_prompt_template": (
            "Please enter your question about the data. I will provide an easy-to-understand answer based on the available information."
        ),
        "few_shot_examples": [
            "Q: What are some key facts about Mars? A: Here are some key facts about Mars based on the available data: [summarized facts from retrieved content].",
            "Q: Tell me something interesting about Jupiter. A: According to the data, Jupiter is known for [relevant fact from knowledge base]."
        ]
    }

    # Domain-specific settings
    DOMAIN_SETTINGS = {
        "domain": "Data Insights / Planetary Information",
        "fallback_response": (
            "I'm sorry, I couldn't find an answer to your question in the available data. Please try rephrasing your question or ask about a different topic."
        ),
        "rag": {
            "enabled": True,
            "retrieval_service": "azure_ai_search",
            "embedding_model": "text-embedding-ada-002",
            "top_k": 5,
            "search_type": "vector_semantic"
        }
    }

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default."""
        return os.getenv(key, default)

    @classmethod
    def validate(cls) -> None:
        """Validate that all required environment variables are set."""
        missing = [k for k in cls.REQUIRED_ENV_VARS if not os.getenv(k)]
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    @classmethod
    def get_api_keys(cls) -> Dict[str, str]:
        """Return a dictionary of required API keys, raising error if missing."""
        keys = {}
        for k in ["AZURE_SEARCH_API_KEY", "AZURE_OPENAI_API_KEY"]:
            v = os.getenv(k)
            if not v:
                raise ConfigError(f"Missing required API key: {k}")
            keys[k] = v
        return keys

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Return LLM configuration, allowing override via environment variables."""
        config = cls.LLM_CONFIG.copy()
        config["model"] = os.getenv("OPENAI_MODEL", config["model"])
        config["temperature"] = float(os.getenv("OPENAI_TEMPERATURE", config["temperature"]))
        config["max_tokens"] = int(os.getenv("OPENAI_MAX_TOKENS", config["max_tokens"]))
        config["system_prompt"] = os.getenv("SYSTEM_PROMPT", config["system_prompt"])
        return config

    @classmethod
    def get_domain_settings(cls) -> Dict[str, Any]:
        """Return domain-specific settings."""
        settings = cls.DOMAIN_SETTINGS.copy()
        # Allow override of fallback response
        settings["fallback_response"] = os.getenv("FALLBACK_RESPONSE", settings["fallback_response"])
        return settings

    @classmethod
    def get_rag_settings(cls) -> Dict[str, Any]:
        """Return RAG pipeline settings."""
        rag = cls.DOMAIN_SETTINGS["rag"].copy()
        rag["embedding_model"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", rag["embedding_model"])
        rag["top_k"] = int(os.getenv("RAG_TOP_K", rag["top_k"]))
        return rag

    @classmethod
    def get_azure_search_config(cls) -> Dict[str, str]:
        """Return Azure AI Search configuration, raising error if missing."""
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        index = os.getenv("AZURE_SEARCH_INDEX_NAME")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        if not endpoint or not index or not api_key:
            raise ConfigError("Missing Azure AI Search configuration (endpoint, index, or API key).")
        return {
            "endpoint": endpoint,
            "index_name": index,
            "api_key": api_key
        }

    @classmethod
    def get_openai_config(cls) -> Dict[str, str]:
        """Return OpenAI configuration, raising error if missing."""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not endpoint or not api_key or not embedding_deployment:
            raise ConfigError("Missing OpenAI configuration (endpoint, API key, or embedding deployment).")
        return {
            "endpoint": endpoint,
            "api_key": api_key,
            "embedding_deployment": embedding_deployment
        }

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Return all configuration as a dictionary, validating required fields."""
        cls.validate()
        config = {
            "llm": cls.get_llm_config(),
            "domain": cls.get_domain_settings(),
            "rag": cls.get_rag_settings(),
            "azure_search": cls.get_azure_search_config(),
            "openai": cls.get_openai_config()
        }
        return config

# Example usage and error handling
def load_config() -> Dict[str, Any]:
    try:
        config = Config.get_all()
        return config
    except ConfigError as e:
        logging.error(f"Configuration error: {e}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected configuration error: {ex}")
        raise

# If this module is run directly, print the loaded config (for debugging)
if __name__ == "__main__":
    try:
        cfg = load_config()
        print("Loaded configuration:")
        for k, v in cfg.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
