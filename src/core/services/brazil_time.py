"""
Brazil timezone utilities for the LangChain application.
"""

from datetime import datetime
from zoneinfo import ZoneInfo


def get_brazil_time() -> datetime:
    """
    Get current time in Brazil timezone (America/Sao_Paulo).

    Android Note: Python's 'datetime' is like 'java.util.Calendar' or
    the newer 'java.time.LocalDateTime'.
    """
    return datetime.now(ZoneInfo("America/Sao_Paulo"))
