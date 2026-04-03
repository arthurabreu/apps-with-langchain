"""
Loading spinner animation for terminal output.
"""

import sys
import threading
import time
from typing import Optional


class LoadingSpinner:
    """Display a loading spinner animation."""

    def __init__(self, message: str = "Loading", delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.busy = False
        self.spinner_thread = None

    def spin(self):
        """Spin the loading animation."""
        while self.busy:
            for char in self.spinner_chars:
                if not self.busy:
                    break
                sys.stdout.write(f'\r{self.message} {char}')
                sys.stdout.flush()
                time.sleep(self.delay)

    def start(self, message: Optional[str] = None):
        """Start the spinner animation."""
        if message:
            self.message = message
        self.busy = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self, final_message: str = ""):
        """Stop the spinner animation."""
        self.busy = False
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)
        # Clear the line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        if final_message:
            print(f"\r{final_message}")
        sys.stdout.flush()
