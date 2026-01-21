"""
Example file showing how to use LangChain with environment variables.
This demonstrates proper API key management for LangChain applications.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def example_openai_usage():
    """
    Example of using OpenAI with LangChain and environment variables.
    """
    try:
        from langchain_openai import ChatOpenAI
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "your-openai-api-key-here":
            print("[X] OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            return
        
        # Initialize ChatOpenAI with environment variable
        llm = ChatOpenAI(
            api_key=api_key,  # This will use the environment variable
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        print("[OK] OpenAI ChatGPT initialized successfully!")
        
        # Example usage
        response = llm.invoke("Hello! How are you?")
        print(f"Response: {response.content}")
        
    except ImportError:
        print("[X] langchain-openai not installed. Run: pip install langchain-openai")
    except Exception as e:
        print(f"[X] Error initializing OpenAI: {e}")

def example_huggingface_usage():
    """
    Example of using Hugging Face with LangChain and environment variables.
    """
    try:
        from langchain_community.llms import HuggingFacePipeline
        
        # Get API key from environment
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if api_key and api_key != "your-huggingface-api-key-here":
            # Set the token for Hugging Face
            os.environ["HUGGINGFACE_HUB_TOKEN"] = api_key
            print("[OK] Hugging Face API key configured!")
        else:
            print("[!] Hugging Face API key not configured. Some models may not be accessible.")
        
        # Example with a local pipeline (doesn't require API key)
        print("[OK] Hugging Face integration ready!")
        
    except ImportError:
        print("[X] langchain-community not installed. Run: pip install langchain-community")
    except Exception as e:
        print(f"[X] Error with Hugging Face: {e}")

def example_environment_based_config():
    """
    Example of using environment variables for application configuration.
    """
    # Get environment settings
    environment = os.getenv("ENVIRONMENT", "production")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"Running in {environment} environment")
    print(f"Debug mode: {debug}")
    
    # Configure based on environment
    if environment == "development":
        print("Development mode: Using verbose logging")
        # Set up development-specific configurations
    elif environment == "production":
        print("Production mode: Optimized for performance")
        # Set up production-specific configurations
    
    return {
        "environment": environment,
        "debug": debug
    }

def main():
    """
    Main function demonstrating various LangChain integrations with environment variables.
    """
    print("LangChain Environment Variable Examples")
    print("=" * 50)
    
    # Show environment configuration
    config = example_environment_based_config()
    print()
    
    # Try OpenAI example
    print("Testing OpenAI Integration:")
    example_openai_usage()
    print()
    
    # Try Hugging Face example
    print("Testing Hugging Face Integration:")
    example_huggingface_usage()
    print()
    
    print("Remember to:")
    print("1. Set your actual API keys in the .env file")
    print("2. Never commit the .env file to version control")
    print("3. Use different .env files for different environments")

if __name__ == "__main__":
    main()