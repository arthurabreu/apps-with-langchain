"""
Dependency injection module.
Provides a simple DI container for managing service lifetimes and dependencies.
"""

from .dependency_injection import DIContainer, get_container, reset_container

__all__ = [
    'DIContainer',
    'get_container',
    'reset_container',
]
