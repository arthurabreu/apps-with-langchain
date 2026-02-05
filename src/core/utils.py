"""
Utility functions for LangChain applications.
"""

def prompt_continue() -> bool:
    """
    Prompt user to continue to next generation or skip.
    
    Returns:
        bool: True to continue, False to skip
    """
    while True:
        choice = input("\nPress Enter to continue to next generation, or 's' to skip: ").strip().lower()
        if choice == "":
            return True
        elif choice == "s":
            print("[INFO] Skipping remaining generations...")
            return False
        else:
            print("Invalid input. Press Enter to continue or 's' to skip.")