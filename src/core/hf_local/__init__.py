"""
Local HuggingFace model support module.
Provides tools for working with local Hugging Face models in LangChain.
"""

from .device_utils import _select_device, _get_max_memory, _build_bnb_config
from .loading_spinner import LoadingSpinner
from .local_hf_model import LocalHuggingFaceModel

__all__ = [
    '_select_device',
    '_get_max_memory',
    '_build_bnb_config',
    'LoadingSpinner',
    'LocalHuggingFaceModel',
]
