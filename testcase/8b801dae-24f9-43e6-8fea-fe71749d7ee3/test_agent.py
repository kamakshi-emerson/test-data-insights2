
import os
import pytest
from unittest.mock import patch

# Assume Config and ConfigError are imported from the module under test
# from mymodule.config import Config, ConfigError

class ConfigError(Exception):
    """Dummy ConfigError for demonstration purposes."""
    pass

class Config:
    """Dummy Config class for demonstration purposes."""
    REQUIRED_VARS = ["API_KEY", "DB_URL", "SECRET"]

    @classmethod
    def validate(cls):
        missing = [var for var in cls.REQUIRED_VARS if not os.environ.get(var)]
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    @classmethod
    def get_all(cls):
        cls.validate()
        return {var: os.environ[var] for var in cls.REQUIRED_VARS}

@pytest.fixture
def required_env_vars():
    """Fixture to provide a dict of all required env vars with dummy values."""
    return {
        "API_KEY": "dummy_api_key",
        "DB_URL": "sqlite:///:memory:",
        "SECRET": "dummy_secret"
    }

def test_integration_configuration_validation_failure(required_env_vars):
    """
    Integration test: Tests that missing required environment variables cause configuration validation to fail with a clear error.
    Unset one or more required environment variables and call Config.validate() or Config.get_all().
    Success criteria:
      - ConfigError is raised
      - Error message contains names of missing variables
    """
    # Set all required env vars first
    with patch.dict(os.environ, required_env_vars, clear=True):
        # Now remove one required variable
        env_copy = required_env_vars.copy()
        env_copy.pop("API_KEY")
        with patch.dict(os.environ, env_copy, clear=True):
            # Test Config.validate()
            with pytest.raises(ConfigError) as excinfo:
                Config.validate()
            assert "API_KEY" in str(excinfo.value)
            assert "Missing required environment variables" in str(excinfo.value)

            # Test Config.get_all()
            with pytest.raises(ConfigError) as excinfo2:
                Config.get_all()
            assert "API_KEY" in str(excinfo2.value)
            assert "Missing required environment variables" in str(excinfo2.value)

    # Test with multiple missing variables
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigError) as excinfo:
            Config.validate()
        for var in Config.REQUIRED_VARS:
            assert var in str(excinfo.value)
        assert "Missing required environment variables" in str(excinfo.value)
