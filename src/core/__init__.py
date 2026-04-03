"""
Core module for the LangChain application.
Provides a clean API following SOLID principles and dependency injection.

Android Analogy:
- This package encapsulates all SOLID-compliant business logic.
- It's provider-agnostic, allowing easy switching between Claude, HuggingFace, etc.
"""

# Main interfaces and data classes
from .interfaces import (
    ILanguageModel,
    ITokenManager,
    IUserInteraction,
    ModelConfig,
    GenerationResult
)

# Exceptions
from .exceptions import (
    LangChainAppError,
    ConfigurationError,
    ModelConfigurationError,
    ApiKeyError,
    GenerationError,
    TokenManagementError,
    UnsupportedProviderError
)

# Services
from .services import (
    ConfigurationManager,
    ApiKeyValidator,
    ConsoleUserInteraction,
    LoggingService
)

# Dependency injection
from .di import get_container, reset_container

# Model implementations
from .models import ModelFactory

# Token management
from .utils import TokenManager

# Backward compatibility utilities
from .utils import prompt_continue, create_claude_model

__all__ = [
    # Interfaces
    'ILanguageModel',
    'ITokenManager',
    'IUserInteraction',
    'ModelConfig',
    'GenerationResult',

    # Exceptions
    'LangChainAppError',
    'ConfigurationError',
    'ModelConfigurationError',
    'ApiKeyError',
    'GenerationError',
    'TokenManagementError',
    'UnsupportedProviderError',

    # Services
    'ConfigurationManager',
    'ApiKeyValidator',
    'ConsoleUserInteraction',
    'LoggingService',
    'TokenManager',
    'ModelFactory',

    # Dependency injection
    'get_container',
    'reset_container',

    # Utilities
    'prompt_continue',
    'create_claude_model'
]
