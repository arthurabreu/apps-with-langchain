"""
Service implementations for the LangChain application.

Android Analogy:
- These are your 'Managers' or 'Repositories'.
- ApiKeyValidator -> Input validation logic.
- ConsoleUserInteraction -> Your View/Presenter for the CLI.
- ConfigurationManager -> SharedPreferences or a remote Config source.
- LoggingService -> A wrapper around Log.d, Log.e, etc., but with file output.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from .interfaces import IApiKeyValidator, IUserInteraction
from .exceptions import ApiKeyError


def get_brazil_time() -> datetime:
    """
    Get current time in Brazil timezone (America/Sao_Paulo).
    
    Android Note: Python's 'datetime' is like 'java.util.Calendar' or 
    the newer 'java.time.LocalDateTime'.
    """
    return datetime.now(ZoneInfo("America/Sao_Paulo"))


class ApiKeyValidator:
    """
    Validates API keys for different providers.
    Analogous to a Validator class in Android (e.g., EmailValidator).
    """
    
    def validate_key(self, api_key: str, provider: str) -> bool:
        """
        Validate an API key for a specific provider.

        Args:
            api_key: The string key to check.
            provider: 'anthropic' or 'huggingface'.

        Returns:
            True if key looks okay.

        Raises:
            ApiKeyError: If validation fails (caught in the UI layer).
        """
        if not api_key:
            raise ApiKeyError(f"API key not provided for {provider}")

        # Check for placeholder values (like developer forgot to set .env)
        placeholder_indicators = ["your-", "placeholder", "example", "test"]
        if any(indicator in api_key.lower() for indicator in placeholder_indicators):
            if not api_key.startswith(("claude-", "hf_")):  # Allow valid prefixes
                raise ApiKeyError(f"API key appears to be a placeholder for {provider}")

        # Basic format validation
        if provider.lower() == "anthropic" and len(api_key) < 20:
            raise ApiKeyError("Anthropic API key appears too short")
        elif provider.lower() == "huggingface" and not api_key.startswith("hf_"):
            # HuggingFace keys are optional for some models
            logging.warning("HuggingFace API key doesn't start with 'hf_' - may be invalid")

        return True


class ConsoleUserInteraction:
    """
    Handles user interaction through console.
    
    Android Analogy: This is like the 'View' implementation in MVP.
    Instead of Toast.makeText or AlertDialog, it uses print() and input().
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Constructor.
        
        Args:
            logger: Optional logger for audit trails.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def prompt_continue(self) -> bool:
        """
        Blocking call that waits for user input.
        Similar to a Modal Dialog in Android.
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
        Show a list of options and return the selection.
        Analogous to a Spinner or a RadioGroup selection.
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
        """Log/Print info level message."""
        print(f"[INFO] {message}")
        if self.logger:
            self.logger.info(message)
    
    def display_error(self, error: str) -> None:
        """Log/Print error level message."""
        print(f"[ERROR] {error}")
        if self.logger:
            self.logger.error(error)
    
    def display_warning(self, message: str) -> None:
        """Log/Print warning level message."""
        print(f"[WARNING] {message}")
        if self.logger:
            self.logger.warning(message)


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


class LoggingService:
    """
    Centralized logging service.
    
    Android Analogy: Timber or a custom Log wrapper.
    It configures how and where logs are written (File + Console).
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """Constructor Injection of ConfigManager."""
        self.config_manager = config_manager
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure the global 'logging' module."""
        log_level = self.config_manager.get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Print to console (Logcat)
                logging.FileHandler('logs/langchain_app.log')  # Write to disk
            ]
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Returns a named logger instance.
        Analogous to 'Log.getTag(name)'.
        """
        return logging.getLogger(name)