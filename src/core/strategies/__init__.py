"""
Generation strategies package.
Contains different strategies for text generation following the Strategy pattern.
"""

from .standard_generation import StandardGenerationStrategy
from .streaming_generation import StreamingGenerationStrategy

__all__ = [
    'StandardGenerationStrategy',
    'StreamingGenerationStrategy'
]