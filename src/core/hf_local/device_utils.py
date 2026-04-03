"""
Device selection and memory management utilities for Hugging Face models.
"""

import os
import torch
import psutil
from typing import Tuple, Dict


def _select_device() -> Tuple[str, torch.dtype]:
    """
    Detect and select the best available device for model inference.

    Returns:
        Tuple of (device_name, dtype)
        - device_name: "cuda", "mps", or "cpu"
        - dtype: torch.float16 for accelerated devices, torch.float32 for CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"[INFO] CUDA detected: Using GPU acceleration (device: {device}, dtype: {dtype})")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"[INFO] MPS detected: Using Apple Silicon acceleration (device: {device}, dtype: {dtype})")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"[INFO] No GPU detected: Using CPU (device: {device}, dtype: {dtype})")

    return device, dtype


def _get_max_memory(fraction: float = 0.75) -> Dict[int, str]:
    """
    Calculate max_memory dict based on CURRENT free memory, not total.
    This is smarter than using a fraction of total, which doesn't account for OS overhead.

    Args:
        fraction: Fraction of CURRENT free memory to use (e.g., 0.75 = 75% of what's free now)

    Returns:
        Dictionary suitable for HuggingFace transformers device_map="auto"
    """
    mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # Get current free VRAM (not total)
            free = torch.cuda.mem_get_info(i)[0] / (1024**2)  # in MiB
            # Use 75% of what's currently free
            mem[i] = f"{int(free * fraction)}MiB"

    # Get current free RAM
    free_ram = psutil.virtual_memory().available / (1024**2)  # in MiB
    mem["cpu"] = f"{int(free_ram * fraction)}MiB"
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
