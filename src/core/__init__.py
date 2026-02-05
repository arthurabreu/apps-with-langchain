"""
Core module for the LangChain application.
Provides a clean API following SOLID principles and dependency injection.
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
from .dependency_injection import get_container, reset_container

# Model implementations
from .models import ModelFactory

# Token management
from .token_utils import TokenManager

# Backward compatibility utilities
from .utils import prompt_continue, create_openai_model, create_claude_model

__all__ = [
    # Interfaces
    'ILanguageModel',
    'ITokenManager', 
    'IUserInteraction',
    'ModelConfig',
    'GenerationResult',
    
    # Exceptions
    'LangChainAppError',
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
    'create_openai_model',
    'create_claude_model'
]