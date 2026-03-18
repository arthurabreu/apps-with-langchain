"""
Main application entry point for the LangChain Model Testing Lab.

Android Analogy:
- This is your 'app' module's 'Application' class or 'MainActivity' launcher.
- It's the bootstrapper that initializes the DI container and starts the UI.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api.deprecation")

import os
import sys

# 0. Fix module resolution (Like setting up your classpath in Gradle)
# This allows imports from 'core' to work even if we run from the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core.config import (
    DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    INTERACTIVE_MAX_TOKENS, RESPONSES_DIR
)
from core.dependency_injection import get_container

# 1. Load environment variables (.env)
# Like reading local.properties or BuildConfig in Android.
load_dotenv()

# 2. Setup Global Constants/Environment
hf_key = os.getenv("HUGGINGFACE_API_KEY")
if hf_key:
    os.environ["HF_TOKEN"] = hf_key


def main():
    """
    Application Entry Point.
    
    Orchestrates the startup:
    1. Initializes DI Container.
    2. Gets the CLI service (The 'Presenter').
    3. Runs the main menu loop.
    """
    # Initialize DI Container (Like Koin.startKoin or Dagger Component init)
    container = get_container()
    
    # Get the orchestrator service
    cli = container.get_cli_service()
    
    # Start the application UI (CLI Menu)
    cli.run_menu()


if __name__ == "__main__":
    # Standard Python idiom: Only run if this file is executed directly.
    # Like the <intent-filter> with action.MAIN in AndroidManifest.xml.
    main()