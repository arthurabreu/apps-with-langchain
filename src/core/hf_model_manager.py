"""
HuggingFace Model Manager - Download and manage models in system-specific folders.
Creates obvious, user-accessible folders for storing HuggingFace models.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple
import platform


def get_hf_models_folder() -> Path:
    """
    Get the system-specific HuggingFace models folder.
    Creates the folder if it doesn't exist.

    Returns:
        Path to the HuggingFace models folder (user-accessible location)

    Examples:
        - Linux/Pop OS: ~/HuggingFaceModels
        - macOS: ~/HuggingFaceModels
        - Windows: ~/Documents/HuggingFaceModels
    """
    system = platform.system()

    if system == "Windows":
        # Windows: Use Documents folder
        hf_folder = Path.home() / "Documents" / "HuggingFaceModels"
    else:
        # Linux (including Pop OS), macOS, etc.: Use home folder
        hf_folder = Path.home() / "HuggingFaceModels"

    # Create folder if it doesn't exist
    hf_folder.mkdir(parents=True, exist_ok=True)

    return hf_folder


def format_size(bytes_size: int) -> str:
    """
    Format bytes to human-readable size.

    Args:
        bytes_size: Size in bytes

    Returns:
        Human-readable size string (e.g., "2.5 GB", "512 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def get_model_size(model_id: str, hf_token: Optional[str] = None) -> Optional[str]:
    """
    Get the download size of a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "MiniMaxAI/MiniMax-M2.1")
        hf_token: Optional HuggingFace API token

    Returns:
        Human-readable size string or None if size cannot be determined
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=hf_token)

        if hasattr(info, 'siblings') and info.siblings:
            # Sum up all file sizes
            total_size = sum(sibling.size for sibling in info.siblings if sibling.size)
            if total_size > 0:
                return format_size(total_size)

        # Fallback to known sizes if API fails
        known_sizes = {
            "mistralai/Mistral-7B-v0.1": "14 GB",
            "mistralai/Mistral-7B-Instruct-v0.1": "14 GB",
            "meta-llama/Llama-2-7b": "13.5 GB",
            "meta-llama/Llama-2-7b-chat-hf": "13.5 GB",
            "MiniMaxAI/MiniMax-M2.1": "450 GB (full) / ~60 GB (quantized)",
            "deepseek-ai/DeepSeek-V3.2": "690 GB (671B parameters - extremely large)",
        }

        if model_id in known_sizes:
            return known_sizes[model_id]

        return None
    except Exception as e:
        # Fallback to known sizes on any error
        known_sizes = {
            "mistralai/Mistral-7B-v0.1": "14 GB",
            "mistralai/Mistral-7B-Instruct-v0.1": "14 GB",
            "meta-llama/Llama-2-7b": "13.5 GB",
            "meta-llama/Llama-2-7b-chat-hf": "13.5 GB",
            "MiniMaxAI/MiniMax-M2.1": "450 GB (full) / ~60 GB (quantized)",
            "deepseek-ai/DeepSeek-V3.2": "690 GB (671B parameters - extremely large)",
        }

        if model_id in known_sizes:
            return known_sizes[model_id]

        return None


def is_model_downloaded(model_id: str, hf_models_folder: Path) -> bool:
    """
    Check if a model is already downloaded in the HuggingFace models folder.

    Args:
        model_id: HuggingFace model ID
        hf_models_folder: Path to HuggingFace models folder

    Returns:
        True if model is downloaded, False otherwise
    """
    # Create a normalized folder name from model ID
    # e.g., "MiniMaxAI/MiniMax-M2.1" -> "MiniMaxAI-MiniMax-M2.1"
    safe_name = model_id.replace("/", "-")
    model_folder = hf_models_folder / safe_name

    # Check if folder exists and has some content
    return model_folder.exists() and any(model_folder.iterdir())


def download_model(
    model_id: str,
    hf_models_folder: Path,
    hf_token: Optional[str] = None,
    show_progress: bool = True
) -> bool:
    """
    Download a HuggingFace model to the specified folder.

    Args:
        model_id: HuggingFace model ID
        hf_models_folder: Path to HuggingFace models folder
        hf_token: Optional HuggingFace API token
        show_progress: Whether to show download progress

    Returns:
        True if download successful, False otherwise
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Create model-specific subfolder
        safe_name = model_id.replace("/", "-")
        model_folder = hf_models_folder / safe_name
        model_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[STEP 1/2] Downloading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
            cache_dir=str(model_folder)
        )
        print(f"[SUCCESS] Tokenizer downloaded to {model_folder}")

        print(f"\n[STEP 2/2] Downloading model {model_id}...")
        print("[INFO] This may take several minutes depending on model size and connection...")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
            token=hf_token,
            cache_dir=str(model_folder),
            device_map="cpu"
        )

        print(f"[SUCCESS] Model downloaded successfully!")
        print(f"[INFO] Location: {model_folder}")

        return True

    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("\n[FIX] Try upgrading transformers:")
        print("  pip install --upgrade transformers torch")
        if "OutputRecorder" in str(e) or "MiniMax" in model_id:
            print("\n[FIX] For MiniMax-M2.1, also try:")
            print("  pip install transformers>=4.36")
        return False

    except Exception as e:
        error_str = str(e)
        print(f"\n[ERROR] Failed to download model: {e}")

        if "OutputRecorder" in error_str:
            print("\n[INFO] This is a transformers compatibility issue.")
            print("[FIX] Upgrade transformers:")
            print("  pip install --upgrade transformers>=4.36")
            print("\n[ALTERNATIVE] Use a different model (Mistral 7B is more stable):")
            print("  1. Run the Android Agent again")
            print("  2. Select Mistral-7B-Instruct (option 1)")
        elif "404" in error_str or "not found" in error_str.lower():
            print(f"\n[ERROR] Model not found: {model_id}")
            print("[FIX] Check the model ID at: https://huggingface.co/models")
        elif "rate limit" in error_str.lower():
            print("\n[ERROR] HuggingFace API rate limit exceeded")
            print("[FIX] Retry in a few minutes, or set HuggingFace token:")
            print("  export HUGGINGFACE_API_KEY=your_token_here")
        elif "disk" in error_str.lower() or "space" in error_str.lower():
            print("\n[ERROR] Not enough disk space")
            print("[FIX] Check available space:")
            print(f"  df -h {hf_models_folder}")
        else:
            print("\n[TIP] If download fails, try manual download:")
            print("  See DOWNLOAD_MODELS.md for instructions")

        return False


def select_hf_model() -> Optional[Tuple[str, Path]]:
    """
    Interactive model selection for HuggingFace models.

    Returns:
        Tuple of (model_id, model_folder) or None if user cancels
    """
    hf_models_folder = get_hf_models_folder()
    hf_token = os.getenv("HUGGINGFACE_API_KEY")

    print("\n" + "=" * 70)
    print("🤖 SELECT HUGGINGFACE MODEL")
    print("=" * 70)
    print(f"\n[INFO] Models will be downloaded to: {hf_models_folder}")
    print()

    # Predefined models (ordered by compatibility and download speed)
    models = {
        "1": {
            "id": "mistralai/Mistral-7B-Instruct-v0.1",
            "description": "Mistral 7B Instruct (RECOMMENDED - stable, fast)",
        },
        "2": {
            "id": "mistralai/Mistral-7B-v0.1",
            "description": "Mistral 7B Base (7B, very fast)",
        },
        "3": {
            "id": "meta-llama/Llama-2-7b-chat-hf",
            "description": "Llama 2 7B Chat (7B, good for chat)",
        },
        "4": {
            "id": "MiniMaxAI/MiniMax-M2.1",
            "description": "MiniMax-M2.1 (⚠️ requires transformers>=4.36, 450GB)",
        },
        "5": {
            "id": "deepseek-ai/DeepSeek-V3.2",
            "description": "DeepSeek V3.2 (⚠️ 671B parameters, 690GB - massive model)",
        },
        "6": {
            "id": "custom",
            "description": "Enter custom HuggingFace model ID",
        }
    }

    print("Available models:")
    for key, model in models.items():
        if model["id"] == "custom":
            print(f"{key}. {model['description']}")
        else:
            # Check if already downloaded
            is_downloaded = is_model_downloaded(model["id"], hf_models_folder)
            status = " ✓ (downloaded)" if is_downloaded else ""

            # Try to get size
            size_info = get_model_size(model["id"], hf_token)
            size_str = f" - {size_info}" if size_info else ""

            print(f"{key}. {model['description']}{size_str}{status}")
    print()

    choice = input("Select model [1-6]: ").strip()

    if choice not in models:
        print("[ERROR] Invalid choice.")
        return None

    if choice == "6":
        # Custom model
        model_id = input("\nEnter HuggingFace model ID (e.g., 'username/model-name'): ").strip()
        if not model_id:
            print("[ERROR] No model ID provided.")
            return None
    else:
        model_id = models[choice]["id"]

    print(f"\n[INFO] Selected model: {model_id}")

    # Check if already downloaded
    if is_model_downloaded(model_id, hf_models_folder):
        print("[INFO] Model is already downloaded!")
        safe_name = model_id.replace("/", "-")
        return model_id, hf_models_folder / safe_name

    # Get model size
    print("\n[INFO] Checking model size...")
    size_info = get_model_size(model_id, hf_token)
    if size_info:
        print(f"[INFO] Download size: {size_info}")
    else:
        print("[INFO] Could not determine exact size (will show during download)")

    # Ask for permission
    print()
    download_choice = input("Download this model? (y/n): ").strip().lower()

    if download_choice != 'y':
        print("[INFO] Download cancelled.")
        return None

    # Download
    success = download_model(model_id, hf_models_folder, hf_token)

    if success:
        safe_name = model_id.replace("/", "-")
        return model_id, hf_models_folder / safe_name
    else:
        return None


def list_downloaded_models() -> list:
    """
    List all models downloaded in the HuggingFace models folder.

    Returns:
        List of model folder paths
    """
    hf_models_folder = get_hf_models_folder()

    if not hf_models_folder.exists():
        return []

    models = []
    for item in hf_models_folder.iterdir():
        if item.is_dir() and any(item.iterdir()):
            models.append(item)

    return sorted(models)


def get_model_folder_size(model_folder: Path) -> str:
    """
    Get the total size of a model folder.

    Args:
        model_folder: Path to model folder

    Returns:
        Human-readable size string
    """
    total = 0
    for path in model_folder.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return format_size(total)
