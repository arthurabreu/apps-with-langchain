"""
Utility functions for LangChain applications.
Provides backward compatibility and convenience functions.
"""

from .dependency_injection import get_container
from .interfaces import ModelConfig


def prompt_continue() -> bool:
    """
    Prompt user to continue to next generation or skip.
    
    Returns:
        bool: True to continue, False to skip
    """
    container = get_container()
    user_interaction = container.get_user_interaction()
    return user_interaction.prompt_continue()


def create_openai_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0.2, max_tokens: int = 512):
    """
    Create an OpenAI model with default configuration.
    Backward compatibility function.
    
    Args:
        model_name: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured OpenAI model instance
    """
    container = get_container()
    factory = container.get_model_factory()
    
    config = ModelConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return factory.create_model("openai", config)


def create_claude_model(model_name: str = "claude-3-haiku-20240307", temperature: float = 0.2, max_tokens: int = 512):
    """
    Create a Claude model with default configuration.
    Backward compatibility function.
    
    Args:
        model_name: Claude model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured Claude model instance
    """
    container = get_container()
    factory = container.get_model_factory()
    
    config = ModelConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return factory.create_model("anthropic", config)