"""
HuggingFace model management module.
Download and manage models in system-specific folders with an interactive selection interface.
"""

from .hf_model_manager import (
    get_hf_models_folder,
    format_size,
    get_model_size,
    resolve_model_path,
    is_model_downloaded,
    download_model,
    select_hf_model,
    list_downloaded_models,
    get_model_folder_size,
)

__all__ = [
    'get_hf_models_folder',
    'format_size',
    'get_model_size',
    'resolve_model_path',
    'is_model_downloaded',
    'download_model',
    'select_hf_model',
    'list_downloaded_models',
    'get_model_folder_size',
]
