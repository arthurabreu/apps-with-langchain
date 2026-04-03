"""
Service implementations for the LangChain application.

Android Analogy:
- These are your 'Managers' or 'Repositories'.
- ApiKeyValidator -> Input validation logic.
- ConsoleUserInteraction -> Your View/Presenter for the CLI.
- ConfigurationManager -> SharedPreferences or a remote Config source.
- LoggingService -> A wrapper around Log.d, Log.e, etc., but with file output.
"""

from .brazil_time import get_brazil_time
from .api_key_validator import ApiKeyValidator
from .configuration_manager import ConfigurationManager
from .console_user_interaction import ConsoleUserInteraction
from .logging_service import LoggingService

__all__ = [
    "get_brazil_time",
    "ApiKeyValidator",
    "ConfigurationManager",
    "ConsoleUserInteraction",
    "LoggingService",
]
