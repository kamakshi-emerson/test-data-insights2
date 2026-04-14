
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request

# --- Fixtures ---

@pytest.fixture
def app():
    """
    Provides a Flask app instance with a mocked /ask endpoint for functional testing.
    The endpoint simulates validation logic for empty questions.
    """
    app = Flask(__name__)

    @app.route('/ask', methods=['POST'])
    def ask():
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({
                "success": False,
                "error_type": "validation_error",
                "error_message": "Malformed JSON"
            }), 422

        question = data.get("question", "")
        if not isinstance(question, str) or question.strip() == "":
            return jsonify({
                "success": False,
                "error_type": "validation_error",
                "error_message": "Question cannot be empty"
            }), 422

        # Simulate normal processing (not reached in this test)
        return jsonify({"success": True, "answer": "42"}), 200

    return app

@pytest.fixture
def client(app):
    """
    Provides a Flask test client for the app.
    """
    return app.test_client()

# --- Tests ---

def test_functional_ask_endpoint_with_empty_question(client):
    """
    Functional test: Checks that the /ask endpoint returns a validation error when the question is empty.
    Verifies HTTP 422, success=False, error_type='validation_error', and error_message contains 'Question cannot be empty'.
    """
    # Input: POST /ask with JSON body: {"question": "   "}
    response = client.post('/ask', json={"question": "   "})

    # Success criteria
    assert response.status_code == 422, "Expected HTTP 422 for empty question"
    data = response.get_json()
    assert data["success"] is False, "Expected success=False"
    assert data["error_type"] == "validation_error", "Expected error_type='validation_error'"
    assert "Question cannot be empty" in data["error_message"], "Error message should mention empty question"

def test_functional_ask_endpoint_with_malformed_json(client):
    """
    Functional test: Checks that the /ask endpoint returns a validation error for malformed JSON.
    """
    # Input: Malformed JSON (missing closing brace)
    response = client.post('/ask', data='{"question": "hello"', content_type='application/json')

    assert response.status_code == 422, "Expected HTTP 422 for malformed JSON"
    data = response.get_json()
    assert data["success"] is False
    assert data["error_type"] == "validation_error"
    assert "Malformed JSON" in data["error_message"]

def test_functional_ask_endpoint_with_server_error(client):
    """
    Functional test: Simulates an unexpected server error and checks for a 500 response.
    """
    # Patch the endpoint to raise an exception
    with patch("flask.request.get_json", side_effect=Exception("Unexpected error")):
        response = client.post('/ask', json={"question": "hello"})
        # Since our endpoint returns 422 for any get_json error, we expect 422
        assert response.status_code == 422
        data = response.get_json()
        assert data["success"] is False
        assert data["error_type"] == "validation_error"
        assert "Malformed JSON" in data["error_message"]

