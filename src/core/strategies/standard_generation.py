"""
Standard generation strategy implementation.
Handles synchronous text generation with token tracking and user interaction.
"""

from typing import Any
from langchain_core.prompts import ChatPromptTemplate

from ..interfaces import IGenerationStrategy, ModelConfig, GenerationResult, GenerationStrategy
from ..exceptions import GenerationError


class StandardGenerationStrategy:
    """Standard synchronous generation strategy."""
    
    def __init__(self, token_manager, user_interaction, logger):
        """
        Initialize the standard generation strategy.
        
        Args:
            token_manager: Token management service
            user_interaction: User interaction service
            logger: Logger instance
        """
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logger = logger
    
    def generate(self, model: Any, prompt: str, config: ModelConfig) -> GenerationResult:
        """
        Generate text using standard synchronous approach.
        
        Args:
            model: The language model instance
            prompt: Input text prompt
            config: Model configuration
            
        Returns:
            GenerationResult with content and metadata
            
        Raises:
            GenerationError: If generation fails
        """
        try:
            # Use system message from config or default
            system_msg = config.system_message or "You are a helpful assistant."
            full_prompt = f"{system_msg}\n{prompt}"
            
            # Token analysis before generation
            prompt_tokens = self.token_manager.count_tokens(full_prompt, config.model_name)
            est_cost = self.token_manager.estimate_cost(prompt_tokens, config.model_name, is_output=False)
            
            self.user_interaction.display_info("Prompt Analysis:")
            self.user_interaction.display_info(f"- Input tokens: {prompt_tokens}")
            self.user_interaction.display_info(f"- Estimated input cost: ${est_cost:.6f}")
            
            # Create prompt template and generate
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("user", "{input}")
            ])
            
            formatted_prompt = prompt_template.format_messages(input=prompt)
            response = model.invoke(formatted_prompt)
            response_text = response.content
            
            # Token analysis after generation
            response_tokens = self.token_manager.count_tokens(response_text, config.model_name)
            response_cost = self.token_manager.estimate_cost(response_tokens, config.model_name, is_output=True)
            
            self.user_interaction.display_info("Response Analysis:")
            self.user_interaction.display_info(f"- Output tokens: {response_tokens}")
            self.user_interaction.display_info(f"- Estimated output cost: ${response_cost:.6f}")
            
            # Log usage
            self.token_manager.log_usage(config.model_name, prompt_tokens, "standard_prompt", is_output=False)
            self.token_manager.log_usage(config.model_name, response_tokens, "standard_response", is_output=True)
            
            # Show summary
            summary = self.token_manager.get_usage_summary()
            self.user_interaction.display_info("Session Summary:")
            self.user_interaction.display_info(f"- Total tokens used: {summary['total_tokens']}")
            self.user_interaction.display_info(f"- Total estimated cost: ${summary['total_cost']:.6f}")
            
            return GenerationResult(
                content=response_text,
                tokens_used=prompt_tokens + response_tokens,
                cost=est_cost + response_cost,
                strategy_used=GenerationStrategy.STANDARD,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "model": config.model_name,
                    "system_message": system_msg
                }
            )
            
        except Exception as e:
            error_msg = f"Standard generation failed: {e}"
            self.logger.error(error_msg)
            raise GenerationError(error_msg)
    
    def supports_model(self, provider: str) -> bool:
        """Check if this strategy supports the given provider."""
        # Standard generation works with all providers
        return True