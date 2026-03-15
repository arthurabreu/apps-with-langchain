"""
Model factory implementation following the Factory pattern.
Creates model instances with proper dependency injection.
"""

import logging
from typing import Dict, Type, Optional, Any
from ..interfaces import ILanguageModel, ITokenManager, IUserInteraction, ModelConfig, IApiKeyValidator, IModelValidator
from ..exceptions import UnsupportedProviderError
from ..services import ConfigurationManager, ApiKeyValidator
from ..config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from .claude_model import ClaudeModel
from .minimax_model import MiniMaxModel


class ModelFactory:
    """Factory for creating language model instances."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        api_key_validator: IApiKeyValidator,
        token_manager: ITokenManager,
        user_interaction: IUserInteraction,
        logging_service,
        cost_tracker: Optional[Any] = None
    ):
        """
        Initialize the model factory.

        Args:
            config_manager: Configuration management service
            api_key_validator: API key validation service
            token_manager: Token management service
            user_interaction: User interaction service
            logging_service: Logging service
            cost_tracker: Cost tracking service (optional)
        """
        self.config_manager = config_manager
        self.api_key_validator = api_key_validator
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logging_service = logging_service
        self.cost_tracker = cost_tracker
        self.logger = logging_service.get_logger(__name__)
        
        # Registry of available model classes
        self._model_registry: Dict[str, Type[ILanguageModel]] = {
            "anthropic": ClaudeModel,
            "minimax": MiniMaxModel,
            "huggingface": MiniMaxModel,  # Alias for convenience
        }
    
    def create_model(self, provider: str, config: ModelConfig) -> ILanguageModel:
        """
        Create a model instance for the specified provider.
        
        Args:
            provider: The model provider (openai, anthropic, minimax, etc.)
            config: Model configuration
            
        Returns:
            Configured model instance
            
        Raises:
            UnsupportedProviderError: If provider is not supported
            ApiKeyError: If API key validation fails
        """
        provider_lower = provider.lower()
        
        if provider_lower not in self._model_registry:
            available = ", ".join(self._model_registry.keys())
            raise UnsupportedProviderError(
                f"Provider '{provider}' not supported. Available providers: {available}"
            )
        
        # Get API key from config or configuration manager (for API-based providers)
        # MiniMax/HuggingFace models don't require API keys, but may use HF token for model download
        if provider_lower != "minimax" and provider_lower != "huggingface":
            if not config.api_key:
                config.api_key = self.config_manager.get_api_key(provider_lower)
            
            # Validate API key for API-based providers
            self.api_key_validator.validate_key(config.api_key, provider_lower)
        
        # Get model class and create instance
        model_class = self._model_registry[provider_lower]
        logger = self.logging_service.get_logger(f"{__name__}.{provider_lower}")
        
        self.logger.info(f"Creating {provider} model: {config.model_name}")

        return model_class(
            config=config,
            token_manager=self.token_manager,
            user_interaction=self.user_interaction,
            logger=logger,
            cost_tracker=self.cost_tracker
        )
    
    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return list(self._model_registry.keys())
    
    def register_model(self, provider: str, model_class: Type[ILanguageModel]) -> None:
        """
        Register a new model class.
        
        Args:
            provider: Provider name
            model_class: Model class to register
        """
        self._model_registry[provider.lower()] = model_class
        self.logger.info(f"Registered model provider: {provider}")
    
    def create_default_claude_model(self, model_name: str = None) -> ILanguageModel:
        """Create a default Claude model with standard configuration."""
        if model_name is None:
            model_name = DEFAULT_MODEL
        config = ModelConfig(
            model_name=model_name,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        return self.create_model("anthropic", config)
    
    def create_minimax_model(self, model_name: str = None, temperature: float = 1.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> ILanguageModel:
        """
        Create a MiniMax-M2.1 model with standard configuration.
        
        Args:
            model_name: Model name (defaults to MiniMax-M2.1)
            temperature: Generation temperature (default: 1.0 as per MiniMax docs)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Configured MiniMax model instance
        """
        if model_name is None:
            model_name = MiniMaxModel.DEFAULT_MODEL_ID
        config = ModelConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self.create_model("minimax", config)
    
    def create_huggingface_model(self, model_name: str, temperature: float = 0.7, max_tokens: int = DEFAULT_MAX_TOKENS) -> ILanguageModel:
        """
        Create a HuggingFace model using MiniMax infrastructure.
        
        Args:
            model_name: HuggingFace model ID
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Configured HuggingFace model instance
        """
        config = ModelConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self.create_model("huggingface", config)