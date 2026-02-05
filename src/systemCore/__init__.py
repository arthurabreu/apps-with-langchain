"""
System Core Package

This package contains the core system architecture and foundation classes.
Think of it like Android's core framework (interfaces, base classes, dependency injection).

Contains:
- Abstract interfaces and contracts
- Exception definitions
- Dependency injection container
- Core services and utilities
"""

from .interfaces import (
    ILanguageModel, 
    ITokenManager, 
    IUserInteraction,
    ModelConfig, 
    GenerationResult,
    GenerationStrategy
)
from .exceptions import (
    LangChainAppError,
    ModelConfigurationError,
    ApiKeyError,
    GenerationError,
    TokenManagementError,
    UnsupportedProviderError
)
from .services import (
    ConfigurationManager,
    ApiKeyValidator,
    ConsoleUserInteraction,
    LoggingService
)
from .dependency_injection import get_container, reset_container
from .utils import prompt_continue, create_openai_model, create_claude_model

__all__ = [
    'ILanguageModel',
    'ITokenManager', 
    'IUserInteraction',
    'ModelConfig',
    'GenerationResult',
    'GenerationStrategy',
    'LangChainAppError',
    'ModelConfigurationError',
    'ApiKeyError',
    'GenerationError',
    'TokenManagementError',
    'UnsupportedProviderError',
    'ConfigurationManager',
    'ApiKeyValidator',
    'ConsoleUserInteraction',
    'LoggingService',
    'get_container',
    'reset_container',
    'prompt_continue',
    'create_openai_model',
    'create_claude_model'
]