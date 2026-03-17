"""
Factory for creating LangChain chat models for tool-calling agents.
Returns raw ChatAnthropic or HuggingFacePipeline instances (not wrapper classes).
"""

import os
import torch
import psutil
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from src.core.config import CODE_TEMPERATURE, INTERACTIVE_MAX_TOKENS


def _get_max_memory(fraction: float = 0.85) -> dict:
    """
    Calculate max_memory dict to cap GPU and CPU memory at fraction of available.

    Args:
        fraction: Fraction of total available memory to use (e.g., 0.85 = 85%)

    Returns:
        Dictionary suitable for HuggingFace transformers device_map="auto"
    """
    mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            mem[i] = f"{int(total * fraction / (1024**2))}MiB"
    ram = psutil.virtual_memory().total
    mem["cpu"] = f"{int(ram * fraction / (1024**2))}MiB"
    return mem


def _build_bnb_config():
    """Build BitsAndBytesConfig for 4-bit quantization."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ValueError("transformers>=4.30 required for BitsAndBytesConfig")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _resolve_hf_cache_path(model_path: str) -> str:
    """
    Resolve a HuggingFace cache directory to the actual snapshot path.

    When models are downloaded with cache_dir, they are stored in a nested
    structure: models--<org>--<name>/snapshots/<hash>/. This function
    finds the actual model files within that structure.

    Args:
        model_path: Path to local model directory

    Returns:
        Resolved path to the actual model files
    """
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return model_path

    # Look for any models--* subdirectory (HuggingFace cache structure)
    for child in model_dir.iterdir():
        if child.is_dir() and child.name.startswith("models--"):
            snapshots_dir = child / "snapshots"
            if snapshots_dir.is_dir():
                snapshots = sorted(snapshots_dir.iterdir())
                if snapshots:
                    resolved = str(snapshots[-1])
                    print(f"[INFO] Resolved HF cache path: {resolved}")
                    return resolved

    return model_path


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
    Build a HuggingFacePipeline instance for tool-calling with GPU + quantization.

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
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
    except ImportError:
        raise ValueError("langchain-huggingface not installed. Install with: pip install langchain-huggingface")

    print("[INFO] Initializing HuggingFace model with GPU acceleration and quantization...")

    # Limit CPU thread usage to 85% of available cores
    cpu_threads = max(1, int(os.cpu_count() * 0.85))
    torch.set_num_threads(cpu_threads)
    print(f"[INFO] CPU threads limited to {cpu_threads}/{os.cpu_count()}")

    resolved_path = _resolve_hf_cache_path(model_path)

    # Setup GPU acceleration and memory limits
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import pipeline as transformers_pipeline

    if torch.cuda.is_available():
        print(f"[INFO] CUDA detected: {torch.cuda.get_device_name(0)}")
        print("[INFO] Applying 4-bit quantization to reduce VRAM usage...")

        # 4-bit quantization config
        bnb_config = _build_bnb_config()

        # Memory limits at 85% of available VRAM and RAM
        max_mem = _get_max_memory(fraction=0.85)
        print(f"[INFO] Memory limits: {max_mem}")

        # Load model with quantization and memory limits
        print("[INFO] Loading model with GPU acceleration...")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_API_KEY")
        )

        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_API_KEY")
        )

        # Create pipeline directly with loaded model and tokenizer
        text_gen_pipeline = transformers_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_new_tokens=max_tokens,
            device_map="auto"
        )

        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    else:
        print("[WARNING] No CUDA GPU detected. Using CPU (slow and memory-intensive).")
        # Fallback to CPU-only
        llm = HuggingFacePipeline.from_model_id(
            model_id=resolved_path,
            task="text-generation",
            pipeline_kwargs={"temperature": temperature, "max_new_tokens": max_tokens}
        )

    return ChatHuggingFace(llm=llm)


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
