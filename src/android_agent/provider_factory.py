"""
Factory for creating LangChain chat models for tool-calling agents.
Returns raw ChatAnthropic or HuggingFacePipeline instances (not wrapper classes).
"""

import os
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from src.core.config import CODE_TEMPERATURE, INTERACTIVE_MAX_TOKENS


def _list_model_files(directory: Path) -> list[str]:
    """
    Glob for *.gguf and *.bin files in a directory.

    Args:
        directory: Path to search

    Returns:
        Sorted list of model file names
    """
    if not directory.is_dir():
        return []

    files = []
    files.extend(f.name for f in directory.glob("*.gguf"))
    files.extend(f.name for f in directory.glob("*.bin"))
    return sorted(files)


def build_claude_chat_model(
    model_name: str,
    temperature: float = CODE_TEMPERATURE,
    max_tokens: int = INTERACTIVE_MAX_TOKENS
) -> ChatAnthropic:
    """
    Build a ChatAnthropic instance for tool-calling.

    Args:
        model_name: Claude model name (e.g., "claude-3-haiku-20240307")
        temperature: Temperature for generation
        max_tokens: Max tokens for response

    Returns:
        ChatAnthropic instance

    Raises:
        ValueError: If API key not found or model config invalid
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


def build_hf_chat_model(
    model_path: str,
    temperature: float = CODE_TEMPERATURE,
    max_tokens: int = INTERACTIVE_MAX_TOKENS
) -> any:
    """
    Build a HuggingFacePipeline instance for tool-calling.

    Args:
        model_path: Path to local model or HuggingFace model ID
        temperature: Temperature for generation
        max_tokens: Max tokens for response

    Returns:
        HuggingFacePipeline instance

    Raises:
        ValueError: If model is GGUF (requires llama-cpp-python) or other issues
    """
    if model_path.endswith(".gguf"):
        raise ValueError(
            "GGUF models require llama-cpp-python. "
            "Install with: pip install llama-cpp-python, then use llama-cpp-python directly."
        )

    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        raise ValueError("langchain-huggingface not installed. Install with: pip install langchain-huggingface")

    print("[WARNING] HuggingFace tool-calling support may be limited. Use Claude API for best results.")

    return HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        pipeline_kwargs={"temperature": temperature, "max_new_tokens": max_tokens}
    )


def get_chat_model(
    provider: str,
    model_path_or_name: str,
    temperature: float = CODE_TEMPERATURE,
    max_tokens: int = INTERACTIVE_MAX_TOKENS
) -> any:
    """
    Get a chat model instance based on provider.

    Args:
        provider: "claude" or "huggingface"
        model_path_or_name: Model name or path
        temperature: Generation temperature
        max_tokens: Max tokens for response

    Returns:
        ChatAnthropic or HuggingFacePipeline instance

    Raises:
        ValueError: If provider unknown or model config invalid
    """
    if provider.lower() == "claude":
        return build_claude_chat_model(model_path_or_name, temperature, max_tokens)
    elif provider.lower() == "huggingface":
        return build_hf_chat_model(model_path_or_name, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")
