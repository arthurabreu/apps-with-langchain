"""
Generation Strategies Package

This package contains different strategies for text generation.
Think of it like Android's strategy pattern implementations (different algorithms for the same task).

Contains:
- Standard text generation strategy
- Streaming text generation strategy
- Strategy interfaces and implementations
"""

from .strategies import StandardGenerationStrategy, StreamingGenerationStrategy

__all__ = [
    'StandardGenerationStrategy',
    'StreamingGenerationStrategy'
]