"""
Model factory implementation following the Factory pattern.

Android Analogy:
- This is a standard Factory class, similar to a ViewModelProvider.Factory 
  or a custom Factory for creating complex objects with many dependencies.
- It uses the Registry pattern to map strings (like 'anthropic') to classes.
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
    """
    Factory for creating language model instances.
    
    This class is responsible for 'wiring' the dependencies (DI) 
    into the model instances it creates.
    """

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
        Initialize the model factory with its own dependencies.
        
        Android Analogy: Constructor Injection.
        """
        self.config_manager = config_manager
        self.api_key_validator = api_key_validator
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logging_service = logging_service
        self.cost_tracker = cost_tracker
        self.logger = logging_service.get_logger(__name__)
        
        # Registry of available model classes
        # Analogous to a Map<String, Class<out ILanguageModel>> in Kotlin.
        self._model_registry: Dict[str, Type[ILanguageModel]] = {
            "anthropic": ClaudeModel,
            "minimax": MiniMaxModel,
            "huggingface": MiniMaxModel,  # Alias for convenience
        }
    
    def create_model(self, provider: str, config: ModelConfig) -> ILanguageModel:
        """
        Create a model instance for the specified provider.
        
        Args:
            provider: The model provider (e.g., 'anthropic').
            config: Model configuration data class.
            
        Returns:
            An implementation of ILanguageModel.
            
        Raises:
            UnsupportedProviderError: If the string doesn't match a registered class.
        """
        provider_lower = provider.lower()
        
        if provider_lower not in self._model_registry:
            available = ", ".join(self._model_registry.keys())
            raise UnsupportedProviderError(
                f"Provider '{provider}' not supported. Available providers: {available}"
            )
        
        # Logic to fetch API keys if not provided
        if provider_lower != "minimax" and provider_lower != "huggingface":
            if not config.api_key:
                config.api_key = self.config_manager.get_api_key(provider_lower)
            
            # Validate API key format
            self.api_key_validator.validate_key(config.api_key, provider_lower)
        
        # Get the class from registry (Dynamic instantiation)
        model_class = self._model_registry[provider_lower]
        logger = self.logging_service.get_logger(f"{__name__}.{provider_lower}")
        
        self.logger.info(f"Creating {provider} model: {config.model_name}")

        # Injecting all necessary services into the new model instance
        return model_class(
            config=config,
            token_manager=self.token_manager,
            user_interaction=self.user_interaction,
            logger=logger,
            cost_tracker=self.cost_tracker
        )
    
    def get_available_providers(self) -> list[str]:
        """Get list of supported provider strings."""
        return list(self._model_registry.keys())
    
    def register_model(self, provider: str, model_class: Type[ILanguageModel]) -> None:
        """
        Register a new model class dynamically.
        Allows 'Open/Closed' principle - you can add new models without changing this class.
        """
        self._model_registry[provider.lower()] = model_class
        self.logger.info(f"Registered model provider: {provider}")
    
    def create_default_claude_model(self, model_name: str = None) -> ILanguageModel:
        """Convenience method to create a Claude model with defaults."""
        if model_name is None:
            model_name = DEFAULT_MODEL
        config = ModelConfig(
            model_name=model_name,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        return self.create_model("anthropic", config)
    
    def create_minimax_model(self, model_name: str = None, temperature: float = 1.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> ILanguageModel:
        """Convenience method to create a MiniMax model with defaults."""
        if model_name is None:
            model_name = MiniMaxModel.DEFAULT_MODEL_ID
        config = ModelConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self.create_model("minimax", config)
    
    def create_huggingface_model(self, model_name: str, temperature: float = 0.7, max_tokens: int = DEFAULT_MAX_TOKENS) -> ILanguageModel:
        """Convenience method to create a HF model with defaults."""
        config = ModelConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self.create_model("huggingface", config)