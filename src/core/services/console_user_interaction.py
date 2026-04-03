"""
Console-based user interaction service for the LangChain application.
"""

import logging


class ConsoleUserInteraction:
    """
    Handles user interaction through console.

    Android Analogy: This is like the 'View' implementation in MVP.
    Instead of Toast.makeText or AlertDialog, it uses print() and input().
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Constructor.

        Args:
            logger: Optional logger for audit trails.
        """
        self.logger = logger or logging.getLogger(__name__)

    def prompt_continue(self) -> bool:
        """
        Blocking call that waits for user input.
        Similar to a Modal Dialog in Android.
        """
        while True:
            choice = input("\nPress Enter to continue to next generation, or 's' to skip: ").strip().lower()
            if choice == "":
                return True
            elif choice == "s":
                self.display_info("Skipping remaining generations...")
                return False
            else:
                self.display_error("Invalid input. Press Enter to continue or 's' to skip.")

    def prompt_choice(self, message: str, choices: list[str]) -> str:
        """
        Show a list of options and return the selection.
        Analogous to a Spinner or a RadioGroup selection.
        """
        while True:
            print(f"\n{message}")
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")

            try:
                selection = input("Enter your choice (number): ").strip()
                index = int(selection) - 1
                if 0 <= index < len(choices):
                    return choices[index]
                else:
                    self.display_error(f"Invalid choice. Please enter a number between 1 and {len(choices)}")
            except ValueError:
                self.display_error("Invalid input. Please enter a number.")

    def display_info(self, message: str) -> None:
        """Log/Print info level message."""
        print(f"[INFO] {message}")
        if self.logger:
            self.logger.info(message)

    def display_error(self, error: str) -> None:
        """Log/Print error level message."""
        print(f"[ERROR] {error}")
        if self.logger:
            self.logger.error(error)

    def display_warning(self, message: str) -> None:
        """Log/Print warning level message."""
        print(f"[WARNING] {message}")
        if self.logger:
            self.logger.warning(message)
