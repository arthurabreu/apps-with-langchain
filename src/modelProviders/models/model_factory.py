"""
Model factory implementation following the Factory pattern.
Creates model instances with proper dependency injection.
"""

import logging
from typing import Dict, Type
from ..interfaces import ILanguageModel, ITokenManager, IUserInteraction, ModelConfig
from ..exceptions import UnsupportedProviderError
from ..services import ConfigurationManager, ApiKeyValidator
from .openai_model import OpenAIModel
from .claude_model import ClaudeModel


class ModelFactory:
    """Factory for creating language model instances."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        api_key_validator: ApiKeyValidator,
        token_manager: ITokenManager,
        user_interaction: IUserInteraction,
        logging_service
    ):
        """
        Initialize the model factory.
        
        Args:
            config_manager: Configuration management service
            api_key_validator: API key validation service
            token_manager: Token management service
            user_interaction: User interaction service
            logging_service: Logging service
        """
        self.config_manager = config_manager
        self.api_key_validator = api_key_validator
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__)
        
        # Registry of available model classes
        self._model_registry: Dict[str, Type[ILanguageModel]] = {
            "openai": OpenAIModel,
            "anthropic": ClaudeModel,
        }
    
    def create_model(self, provider: str, config: ModelConfig) -> ILanguageModel:
        """
        Create a model instance for the specified provider.
        
        Args:
            provider: The model provider (openai, anthropic, etc.)
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
        
        # Get API key from config or configuration manager
        if not config.api_key:
            config.api_key = self.config_manager.get_api_key(provider_lower)
        
        # Validate API key
        self.api_key_validator.validate_key(config.api_key, provider_lower)
        
        # Get model class and create instance
        model_class = self._model_registry[provider_lower]
        logger = self.logging_service.get_logger(f"{__name__}.{provider_lower}")
        
        self.logger.info(f"Creating {provider} model: {config.model_name}")
        
        return model_class(
            config=config,
            token_manager=self.token_manager,
            user_interaction=self.user_interaction,
            logger=logger
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
    
    def create_default_openai_model(self, model_name: str = "gpt-3.5-turbo") -> ILanguageModel:
        """Create a default OpenAI model with standard configuration."""
        config = ModelConfig(
            model_name=model_name,
            temperature=0.2,
            max_tokens=512
        )
        return self.create_model("openai", config)
    
    def create_default_claude_model(self, model_name: str = "claude-3-haiku-20240307") -> ILanguageModel:
        """Create a default Claude model with standard configuration."""
        config = ModelConfig(
            model_name=model_name,
            temperature=0.2,
            max_tokens=512
        )
        return self.create_model("anthropic", config)