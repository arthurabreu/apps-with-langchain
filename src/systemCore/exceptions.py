"""
Custom exceptions for the LangChain application.
Provides specific error types for better error handling and debugging.
Follows a hierarchical structure for better exception handling.
"""

from typing import Optional, Dict, Any


class LangChainAppError(Exception):
    """
    Base exception for all application errors.
    
    Attributes:
        message: Error message
        error_code: Optional error code for categorization
        context: Optional context information
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# Configuration-related exceptions
class ConfigurationError(LangChainAppError):
    """Base class for configuration-related errors."""
    pass


class ModelConfigurationError(ConfigurationError):
    """Raised when model configuration is invalid."""
    pass


class ApiKeyError(ConfigurationError):
    """Raised when API key is missing or invalid."""
    pass


class EnvironmentConfigurationError(ConfigurationError):
    """Raised when environment configuration is invalid."""
    pass


# Model-related exceptions
class ModelError(LangChainAppError):
    """Base class for model-related errors."""
    pass


class ModelInitializationError(ModelError):
    """Raised when model fails to initialize."""
    pass


class UnsupportedProviderError(ModelError):
    """Raised when an unsupported model provider is requested."""
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass


# Generation-related exceptions
class GenerationError(LangChainAppError):
    """Base class for text generation errors."""
    pass


class PromptError(GenerationError):
    """Raised when prompt processing fails."""
    pass


class StreamingError(GenerationError):
    """Raised when streaming generation fails."""
    pass


class TokenLimitExceededError(GenerationError):
    """Raised when token limit is exceeded."""
    pass


class RateLimitError(GenerationError):
    """Raised when API rate limit is exceeded."""
    pass


# Token management exceptions
class TokenManagementError(LangChainAppError):
    """Base class for token management errors."""
    pass


class TokenCountingError(TokenManagementError):
    """Raised when token counting fails."""
    pass


class CostEstimationError(TokenManagementError):
    """Raised when cost estimation fails."""
    pass


class UsageTrackingError(TokenManagementError):
    """Raised when usage tracking fails."""
    pass


# Service-related exceptions
class ServiceError(LangChainAppError):
    """Base class for service-related errors."""
    pass


class DependencyInjectionError(ServiceError):
    """Raised when dependency injection fails."""
    pass


class ServiceNotFoundError(ServiceError):
    """Raised when a required service is not found."""
    pass


class ServiceInitializationError(ServiceError):
    """Raised when service initialization fails."""
    pass


# User interaction exceptions
class UserInteractionError(LangChainAppError):
    """Base class for user interaction errors."""
    pass


class UserCancelledError(UserInteractionError):
    """Raised when user cancels an operation."""
    pass


class InvalidUserInputError(UserInteractionError):
    """Raised when user provides invalid input."""
    pass


# Strategy-related exceptions
class StrategyError(LangChainAppError):
    """Base class for strategy-related errors."""
    pass


class UnsupportedStrategyError(StrategyError):
    """Raised when an unsupported strategy is requested."""
    pass


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""
    pass