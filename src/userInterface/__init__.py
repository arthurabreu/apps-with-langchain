"""
User Interface Package

This package handles all user interaction and interface logic.
Think of it like Android's UI layer (Activities, Fragments, ViewModels).

Contains:
- Main application entry point
- Console interface and menu system
- User interaction utilities
- Test scripts
"""

from .main import main, print_api_key_status, check_memory_usage

__all__ = [
    'main',
    'print_api_key_status', 
    'check_memory_usage'
]