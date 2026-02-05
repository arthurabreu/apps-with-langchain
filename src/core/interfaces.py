"""
Abstract interfaces and base classes for the LangChain application.
Defines contracts for models, token managers, and other core components.
Follows Interface Segregation Principle with focused, specific interfaces.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Protocol
from dataclasses import dataclass
from enum import Enum


class GenerationStrategy(Enum):
    """Strategies for text generation."""
    STANDARD = "standard"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class ModelConfig:
    """Configuration for language models."""
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 512
    api_key: Optional[str] = None
    system_message: Optional[str] = None
    generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Result of a text generation operation."""
    content: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    strategy_used: Optional[GenerationStrategy] = None


# Segregated Token Management Interfaces
class ITokenCounter(Protocol):
    """Interface for token counting operations."""
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text for a specific model."""
        ...


class ICostEstimator(Protocol):
    """Interface for cost estimation operations."""
    
    def estimate_cost(self, tokens: int, model_name: str, is_output: bool = False) -> float:
        """Estimate cost for token usage."""
        ...


class IUsageTracker(Protocol):
    """Interface for usage tracking operations."""
    
    def log_usage(self, model_name: str, tokens: int, operation_type: str, is_output: bool = False) -> None:
        """Log token usage."""
        ...
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        ...


class ITokenManager(ITokenCounter, ICostEstimator, IUsageTracker, Protocol):
    """Combined interface for token management operations."""
    pass


# Segregated User Interaction Interfaces
class IUserPrompt(Protocol):
    """Interface for user prompting operations."""
    
    def prompt_continue(self) -> bool:
        """Prompt user to continue operation."""
        ...
    
    def prompt_choice(self, message: str, choices: list[str]) -> str:
        """Prompt user to make a choice."""
        ...


class IUserDisplay(Protocol):
    """Interface for user display operations."""
    
    def display_info(self, message: str) -> None:
        """Display information to user."""
        ...
    
    def display_error(self, error: str) -> None:
        """Display error to user."""
        ...
    
    def display_warning(self, message: str) -> None:
        """Display warning to user."""
        ...


class IUserInteraction(IUserPrompt, IUserDisplay, Protocol):
    """Combined interface for user interaction operations."""
    pass


# Model-related Interfaces
class IModelValidator(Protocol):
    """Interface for model configuration validation."""
    
    def validate_config(self, config: ModelConfig) -> None:
        """Validate model configuration."""
        ...


class IApiKeyValidator(Protocol):
    """Interface for API key validation."""
    
    def validate_key(self, api_key: str, provider: str) -> bool:
        """Validate an API key for a specific provider."""
        ...


class IGenerationStrategy(Protocol):
    """Interface for text generation strategies."""
    
    def generate(self, model: Any, prompt: str, config: ModelConfig) -> GenerationResult:
        """Generate text using this strategy."""
        ...
    
    def supports_model(self, provider: str) -> bool:
        """Check if this strategy supports the given provider."""
        ...


class ILanguageModel(ABC):
    """Abstract base class for language models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the model configuration."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the model provider name."""
        pass


class IModelFactory(Protocol):
    """Interface for model factory."""
    
    def create_model(self, provider: str, config: ModelConfig) -> ILanguageModel:
        """Create a model instance for the specified provider."""
        ...
    
    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        ...
    
    def register_model(self, provider: str, model_class: type) -> None:
        """Register a new model provider."""
        ...