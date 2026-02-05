"""
Claude model implementation following SOLID principles.
Refactored to use dependency injection and proper separation of concerns.
"""

import logging
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from ..interfaces import ILanguageModel, ITokenManager, IUserInteraction, ModelConfig, GenerationResult
from ..exceptions import ModelConfigurationError, ApiKeyError, GenerationError


class ClaudeModel(ILanguageModel):
    """Claude model implementation with dependency injection."""
    
    def __init__(
        self, 
        config: ModelConfig,
        token_manager: ITokenManager,
        user_interaction: IUserInteraction,
        logger: logging.Logger
    ):
        """
        Initialize Claude model with dependencies.
        
        Args:
            config: Model configuration
            token_manager: Token management service
            user_interaction: User interaction service
            logger: Logger instance
        """
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logger = logger
        self._model = None
        
        super().__init__(config)
        self._initialize_model()
    
    def _validate_config(self) -> None:
        """Validate the model configuration."""
        if not self.config.api_key:
            raise ApiKeyError("Anthropic API key not provided")
        
        if not self.config.model_name:
            raise ModelConfigurationError("Model name not provided")
        
        if not (0.0 <= self.config.temperature <= 1.0):
            raise ModelConfigurationError("Temperature must be between 0.0 and 1.0")
        
        if self.config.max_tokens <= 0:
            raise ModelConfigurationError("Max tokens must be positive")
    
    def _initialize_model(self) -> None:
        """Initialize the Claude model."""
        try:
            self.logger.info(f"Initializing Claude model: {self.config.model_name}")
            self.logger.info(f"Temperature: {self.config.temperature}")
            self.logger.info(f"Max tokens: {self.config.max_tokens}")
            
            self._model = ChatAnthropic(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key
            )
            
            self.logger.info("Claude model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude model: {e}")
            raise ModelConfigurationError(f"Failed to initialize Claude model: {e}")
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with content and metadata
            
        Raises:
            GenerationError: If generation fails
        """
        try:
            # Check if user wants to continue
            skip_prompt = kwargs.get('skip_prompt', False)
            if not skip_prompt and not self.user_interaction.prompt_continue():
                self.user_interaction.display_info("Generation skipped by user.")
                return GenerationResult(content="Generation skipped by user.")
            
            self.user_interaction.display_info(f"Prompt: {prompt[:100]}...")
            self.user_interaction.display_info("Generating using Claude...")
            
            # Use the appropriate generation strategy
            from ..strategies.standard_generation import StandardGenerationStrategy
            from ..strategies.streaming_generation import StreamingGenerationStrategy
            
            if self.config.generation_strategy.value == "streaming":
                strategy = StreamingGenerationStrategy(
                    self.token_manager, 
                    self.user_interaction, 
                    self.logger
                )
            else:
                strategy = StandardGenerationStrategy(
                    self.token_manager, 
                    self.user_interaction, 
                    self.logger
                )
            
            # Generate using the selected strategy
            result = strategy.generate(self._model, prompt, self.config)
            
            self.user_interaction.display_info("Generation complete!")
            return result
            
        except Exception as e:
            error_msg = f"Generation failed: {e}"
            self.logger.error(error_msg)
            self.user_interaction.display_error(error_msg)
            raise GenerationError(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": self.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "status": "ready" if self._model else "not_initialized"
        }
    
    @property
    def provider(self) -> str:
        """Get the model provider name."""
        return "Anthropic"