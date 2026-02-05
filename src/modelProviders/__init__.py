"""
Model Providers Package

This package handles all AI model integrations and providers.
Think of it like Android's service providers or API clients (Retrofit, OkHttp).

Contains:
- Model implementations (OpenAI, Claude, HuggingFace)
- Model factory for creating instances
- Local model handling utilities
"""

from .models import ModelFactory, OpenAIModel, ClaudeModel
from .langchain_huggingface_local import LocalHuggingFaceModel, LoadingSpinner

__all__ = [
    'ModelFactory',
    'OpenAIModel',
    'ClaudeModel',
    'LocalHuggingFaceModel',
    'LoadingSpinner'
]