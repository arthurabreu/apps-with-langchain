"""
Service implementations for the LangChain application.
Contains concrete implementations of interfaces for dependency injection.
"""

import os
import logging
from typing import Dict, Any
from .interfaces import IApiKeyValidator, IUserInteraction
from .exceptions import ApiKeyError


class ApiKeyValidator:
    """Validates API keys for different providers."""
    
    def validate_key(self, api_key: str, provider: str) -> bool:
        """
        Validate an API key for a specific provider.
        
        Args:
            api_key: The API key to validate
            provider: The provider name (openai, anthropic, etc.)
            
        Returns:
            True if key appears valid, False otherwise
            
        Raises:
            ApiKeyError: If key is missing or obviously invalid
        """
        if not api_key:
            raise ApiKeyError(f"API key not provided for {provider}")
        
        # Check for placeholder values
        placeholder_indicators = ["your-", "sk-", "placeholder", "example", "test"]
        if any(indicator in api_key.lower() for indicator in placeholder_indicators):
            if not api_key.startswith(("sk-", "claude-", "hf_")):  # Allow valid prefixes
                raise ApiKeyError(f"API key appears to be a placeholder for {provider}")
        
        # Basic format validation
        if provider.lower() == "openai" and not api_key.startswith("sk-"):
            raise ApiKeyError("OpenAI API key should start with 'sk-'")
        elif provider.lower() == "anthropic" and len(api_key) < 20:
            raise ApiKeyError("Anthropic API key appears too short")
        elif provider.lower() == "huggingface" and not api_key.startswith("hf_"):
            # HuggingFace keys are optional for some models
            logging.warning("HuggingFace API key doesn't start with 'hf_' - may be invalid")
        
        return True


class ConsoleUserInteraction:
    """Handles user interaction through console."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def prompt_continue(self) -> bool:
        """
        Prompt user to continue operation.
        
        Returns:
            True to continue, False to skip
        """
        while True:
            choice = input("\nPress Enter to continue to next generation, or 's' to skip: ").strip().lower()
            if choice == "":
                return True
            elif choice == "s":
                self.display_info("Skipping remaining generations...")
                return False
            else:
                self.display_error("Invalid input. Press Enter to continue or 's' to skip.")
    
    def prompt_choice(self, message: str, choices: list[str]) -> str:
        """
        Prompt user to make a choice from available options.
        
        Args:
            message: The prompt message
            choices: List of available choices
            
        Returns:
            The selected choice
        """
        while True:
            print(f"\n{message}")
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")
            
            try:
                selection = input("Enter your choice (number): ").strip()
                index = int(selection) - 1
                if 0 <= index < len(choices):
                    return choices[index]
                else:
                    self.display_error(f"Invalid choice. Please enter a number between 1 and {len(choices)}")
            except ValueError:
                self.display_error("Invalid input. Please enter a number.")
    
    def display_info(self, message: str) -> None:
        """Display information message."""
        print(f"[INFO] {message}")
        if self.logger:
            self.logger.info(message)
    
    def display_error(self, error: str) -> None:
        """Display error message."""
        print(f"[ERROR] {error}")
        if self.logger:
            self.logger.error(error)
    
    def display_warning(self, message: str) -> None:
        """Display warning message."""
        print(f"[WARNING] {message}")
        if self.logger:
            self.logger.warning(message)


class ConfigurationManager:
    """Manages application configuration."""
    
    def __init__(self):
        self._config = {}
        self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        self._config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        key_map = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "huggingface": "huggingface_api_key",
            "google": "google_api_key",
        }
        
        config_key = key_map.get(provider.lower())
        if not config_key:
            raise ValueError(f"Unknown provider: {provider}")
        
        return self.get(config_key)


class LoggingService:
    """Centralized logging service."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config_manager.get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('langchain_app.log')
            ]
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)