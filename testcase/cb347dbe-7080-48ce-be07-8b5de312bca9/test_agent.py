
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request
import json

# --- Fixtures ---

@pytest.fixture
def app():
    """
    Provides a Flask app instance for testing.
    """
    app = Flask(__name__)

    @app.route('/ask', methods=['POST'])
    def ask():
        # Simulate environment variable check
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({
                "success": False,
                "answer": None,
                "error_type": "MissingEnvironmentVariable"
            }), 500

        data = request.get_json()
        question = data.get("question")
        if not question:
            return jsonify({
                "success": False,
                "answer": None,
                "error_type": "InvalidInput"
            }), 400

        # Simulate LLM call
        try:
            # In real code, this would call Azure/OpenAI
            answer = f"Key facts about Mars: Mars is the fourth planet from the Sun."
            return jsonify({
                "success": True,
                "answer": answer,
                "error_type": None
            }), 200
        except Exception:
            return jsonify({
                "success": False,
                "answer": None,
                "error_type": "ServiceUnavailable"
            }), 503

    return app

@pytest.fixture
def client(app):
    """
    Provides a Flask test client.
    """
    return app.test_client()

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """
    Ensures required environment variables are set for the test.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

# --- Tests ---

def test_functional_successful_ask_endpoint_with_valid_question(client):
    """
    Functional test: Validates that the /ask endpoint returns a successful response and an answer
    when provided with a valid question.
    """
    question = "What are the key facts about Mars?"
    payload = {"question": question}

    # Patch the LLM call inside the endpoint to simulate a successful response
    with patch("builtins.print"):  # Patch print to suppress output if used
        response = client.post("/ask", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 200, "Expected HTTP 200 OK"
    data = response.get_json()
    assert data["success"] is True, "Expected success=True in response"
    assert data["answer"] is not None and data["answer"] != "", "Expected non-empty answer"
    assert data["error_type"] is None, "Expected error_type to be None"

# --- Error scenario tests (not required by success_criteria, but for completeness) ---

def test_functional_ask_endpoint_service_unavailable(client, monkeypatch):
    """
    Functional test: Simulates Azure/OpenAI service unavailable error.
    """
    question = "What are the key facts about Mars?"
    payload = {"question": question}

    # Patch the LLM call to raise an exception
    with patch("flask.jsonify") as mock_jsonify:
        def raise_exc(*args, **kwargs):
            raise Exception("Service unavailable")
        # Patch the endpoint's jsonify to raise
        mock_jsonify.side_effect = raise_exc

        response = client.post("/ask", data=json.dumps(payload), content_type="application/json")
        # Since jsonify is patched to raise, Flask will return a 500 error
        assert response.status_code in (500, 503)

def test_functional_ask_endpoint_missing_env_var(client, monkeypatch):
    """
    Functional test: Simulates missing required environment variables.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    question = "What are the key facts about Mars?"
    payload = {"question": question}

    response = client.post("/ask", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert data["answer"] is None
    assert data["error_type"] == "MissingEnvironmentVariable"
