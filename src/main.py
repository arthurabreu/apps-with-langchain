"""
Main application file for LangChain project.
This file demonstrates how to use environment variables with python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function demonstrating environment variable usage.
    """
    print("LangChain Environment Setup")
    print("=" * 40)
    
    # Get API keys from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Check which API keys are configured
    print("API Key Status:")
    print(f"OpenAI: {'[OK] Configured' if openai_key and openai_key != 'your-openai-api-key-here' else '[X] Not configured'}")
    print(f"Google: {'[OK] Configured' if google_key and google_key != 'your-google-api-key-here' else '[X] Not configured'}")
    print(f"Hugging Face: {'[OK] Configured' if huggingface_key and huggingface_key != 'your-huggingface-api-key-here' else '[X] Not configured'}")
    print(f"Anthropic: {'[OK] Configured' if anthropic_key and anthropic_key != 'your-anthropic-api-key-here' else '[X] Not configured'}")
    
    # Environment settings
    environment = os.getenv("ENVIRONMENT", "production")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"\nEnvironment: {environment}")
    print(f"Debug Mode: {debug}")
    
    print("\nTo configure your API keys:")
    print("1. Edit the .env file in the project root")
    print("2. Replace 'your-api-key-here' with your actual API keys")
    print("3. Never commit the .env file to version control!")

if __name__ == "__main__":
    main()