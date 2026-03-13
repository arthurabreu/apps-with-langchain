"""
Streaming generation strategy implementation.
Handles streaming text generation for real-time output.
"""

from typing import Any

from ..interfaces import IGenerationStrategy, ModelConfig, GenerationResult, GenerationStrategy
from ..exceptions import GenerationError


class StreamingGenerationStrategy:
    """Streaming generation strategy for real-time output."""

    def __init__(self, token_manager, user_interaction, logger, cost_tracker=None):
        """
        Initialize the streaming generation strategy.

        Args:
            token_manager: Token management service
            user_interaction: User interaction service
            logger: Logger instance
            cost_tracker: Cost tracking service (optional)
        """
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logger = logger
        self.cost_tracker = cost_tracker

    def generate(self, model: Any, prompt: str, config: ModelConfig, context_key: str = "none") -> GenerationResult:
        """
        Generate text using streaming approach.

        Args:
            model: The language model instance
            prompt: Input text prompt
            config: Model configuration
            context_key: The context key (for cost tracking)

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

            self.user_interaction.display_info("Streaming Generation:")
            self.user_interaction.display_info(f"- Input tokens: {prompt_tokens}")
            self.user_interaction.display_info(f"- Estimated input cost: ${est_cost:.6f}")
            self.user_interaction.display_info("- Starting stream...")

            # Create messages directly to avoid template variable substitution issues
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]

            # Stream the response
            response_chunks = []
            print("\n[STREAMING] ", end="", flush=True)

            try:
                # Try streaming if supported
                for chunk in model.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)
                        response_chunks.append(chunk.content)
            except AttributeError:
                # Fallback to regular generation if streaming not supported
                self.user_interaction.display_warning("Streaming not supported, falling back to standard generation")
                response = model.invoke(messages)
                response_text = response.content
                print(response_text)
                response_chunks = [response_text]

            print("\n")  # New line after streaming

            # Combine all chunks
            response_text = "".join(response_chunks)

            # Token analysis after generation
            response_tokens = self.token_manager.count_tokens(response_text, config.model_name)
            response_cost = self.token_manager.estimate_cost(response_tokens, config.model_name, is_output=True)

            self.user_interaction.display_info("Stream Complete:")
            self.user_interaction.display_info(f"- Output tokens: {response_tokens}")
            self.user_interaction.display_info(f"- Estimated output cost: ${response_cost:.6f}")

            # Log usage
            self.token_manager.log_usage(config.model_name, prompt_tokens, "streaming_prompt", is_output=False)
            self.token_manager.log_usage(config.model_name, response_tokens, "streaming_response", is_output=True)

            # Log to cost tracker if available
            if self.cost_tracker:
                self.cost_tracker.log(
                    model=config.model_name,
                    input_tokens=prompt_tokens,
                    output_tokens=response_tokens,
                    source="api_app",
                    context=context_key,
                    prompt_preview=prompt[:100]
                )

            return GenerationResult(
                content=response_text,
                tokens_used=prompt_tokens + response_tokens,
                cost=est_cost + response_cost,
                strategy_used=GenerationStrategy.STREAMING,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "model": config.model_name,
                    "system_message": system_msg,
                    "chunks_count": len(response_chunks)
                }
            )
            
        except Exception as e:
            error_msg = f"Streaming generation failed: {e}"
            self.logger.error(error_msg)
            raise GenerationError(error_msg)
    
    def supports_model(self, provider: str) -> bool:
        """Check if this strategy supports the given provider."""
        # Most modern providers support streaming
        supported_providers = ["anthropic"]
        return provider.lower() in supported_providers