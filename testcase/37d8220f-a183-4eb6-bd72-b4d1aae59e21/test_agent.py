
import os
import pytest
from unittest.mock import patch, MagicMock

# Assume Config is imported from the module under test
# from my_module import Config

class DummyOpenAIClient:
    pass

class DummySearchClient:
    pass

@pytest.fixture
def openai_env_vars():
    """Fixture to set OpenAI environment variables."""
    env = {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_API_TYPE": "azure",
        "OPENAI_API_VERSION": "2023-05-15"
    }
    with patch.dict(os.environ, env):
        yield env

@pytest.fixture
def search_env_vars():
    """Fixture to set Azure Search environment variables."""
    env = {
        "AZURE_SEARCH_SERVICE": "test-search-service",
        "AZURE_SEARCH_KEY": "test-search-key",
        "AZURE_SEARCH_INDEX": "test-search-index"
    }
    with patch.dict(os.environ, env):
        yield env

@pytest.fixture
def config_class():
    """Fixture to provide the Config class with patched client creation methods."""
    # Dummy Config class for demonstration; replace with actual import in real tests
    class Config:
        @staticmethod
        def get_openai_client():
            # Simulate environment variable checks
            required = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_TYPE", "OPENAI_API_VERSION"]
            for var in required:
                if not os.environ.get(var):
                    raise RuntimeError(f"Missing environment variable: {var}")
            # Simulate client creation
            return DummyOpenAIClient()

        @staticmethod
        def get_search_client():
            required = ["AZURE_SEARCH_SERVICE", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX"]
            for var in required:
                if not os.environ.get(var):
                    raise RuntimeError(f"Missing environment variable: {var}")
            return DummySearchClient()
    return Config

def test_integration_openai_and_azure_search_client_creation(openai_env_vars, search_env_vars, config_class):
    """
    Integration test for Config.get_openai_client() and Config.get_search_client().
    Verifies that clients are created when all required environment variables are set,
    and RuntimeError is raised when any variable is missing.
    """
    Config = config_class

    # All variables set: should succeed
    openai_client = Config.get_openai_client()
    assert isinstance(openai_client, DummyOpenAIClient), "Expected DummyOpenAIClient instance when env vars are set"

    search_client = Config.get_search_client()
    assert isinstance(search_client, DummySearchClient), "Expected DummySearchClient instance when env vars are set"

    # Unset one OpenAI variable: should raise RuntimeError
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        with pytest.raises(RuntimeError) as excinfo:
            Config.get_openai_client()
        assert "Missing environment variable: OPENAI_API_KEY" in str(excinfo.value)

    # Unset one Search variable: should raise RuntimeError
    with patch.dict(os.environ, {"AZURE_SEARCH_KEY": ""}):
        with pytest.raises(RuntimeError) as excinfo:
            Config.get_search_client()
        assert "Missing environment variable: AZURE_SEARCH_KEY" in str(excinfo.value)

    # Error scenario: Unexpected exception type should not occur
    # (If it does, the test will fail naturally)

