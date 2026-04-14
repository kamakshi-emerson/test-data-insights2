
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

@pytest.fixture
def client():
    """
    Fixture to provide a test client for the FastAPI/Flask app.
    Assumes the app is imported as 'app' from 'main.py'.
    """
    try:
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)
    except ImportError:
        # Fallback for Flask
        from flask import Flask
        from main import app
        return app.test_client()

def test_functional_malformed_json_in_ask_endpoint(client):
    """
    Functional test: Checks that the /ask endpoint returns a 400 error with appropriate message when given malformed JSON.
    """
    # Prepare malformed JSON (missing closing brace)
    malformed_json = '{"question": "What is the weather today?"'
    headers = {'Content-Type': 'application/json'}

    # Patch any external HTTP calls inside the endpoint to prevent real network connections
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {})

        # Send POST request with malformed JSON
        response = client.post("/ask", data=malformed_json, headers=headers)

    assert response.status_code == 400, "Expected HTTP 400 for malformed JSON"

    # Try to parse JSON response, fallback to text if not possible
    try:
        resp_json = response.json()
    except Exception:
        import json
        resp_json = json.loads(response.data.decode())

    assert resp_json.get("success") is False, "Expected success=False in response"
    assert resp_json.get("error_type") == "malformed_json", "Expected error_type='malformed_json'"
    assert "Malformed JSON" in resp_json.get("error_message", ""), "Error message should mention 'Malformed JSON'"
