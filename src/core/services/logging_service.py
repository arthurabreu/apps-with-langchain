"""
Centralized logging service for the LangChain application.
"""

import logging

from .configuration_manager import ConfigurationManager


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
                logging.FileHandler('src/logs/langchain_app.log')  # Write to disk
            ]
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Returns a named logger instance.
        Analogous to 'Log.getTag(name)'.
        """
        return logging.getLogger(name)
