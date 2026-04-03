"""
Configuration management service for the LangChain application.
"""

import os
from typing import Any


class ConfigurationManager:
    """
    Manages application configuration.

    Android Analogy: SharedPreferences or a wrapper around BuildConfig/Local Properties.
    """

    def __init__(self):
        """Initializes by loading from environment variables (.env file)."""
        self._config = {}
        self._load_from_env()

    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        Analogous to reading from local.properties.
        """
        self._config = {
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value (In-memory only for this session)."""
        self._config[key] = value

    def get_api_key(self, provider: str) -> str:
        """Helper to get a specific provider's API key."""
        key_map = {
            "anthropic": "anthropic_api_key",
            "huggingface": "huggingface_api_key",
        }

        config_key = key_map.get(provider.lower())
        if not config_key:
            raise ValueError(f"Unknown provider: {provider}")

        return self.get(config_key)
