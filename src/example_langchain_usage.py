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
        model = "gpt-3.5-turbo"
        
        if not api_key or api_key == "your-openai-api-key-here":
            print("[X] OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            return
        
        # Initialize ChatOpenAI with environment variable
        llm = ChatOpenAI(
            api_key=api_key,  # This will use the environment variable
            model=model,
            temperature=0.7
        )
        
        print("[OK] OpenAI ChatGPT initialized successfully!")
        print(f"Using model: {model}")
        
    except ImportError:
        print("[X] langchain-openai not installed. Run: pip install langchain-openai")
    except Exception as e:
        print(f"[X] Error initializing OpenAI: {e}")

def example_claude_usage():
    """
    Example of using Anthropic Claude with LangChain and environment variables.
    """
    try:
        # Correct imports for split LangChain packages
        from langchain_anthropic import ChatAnthropic
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            from langchain.schema import HumanMessage, SystemMessage
        from token_utils import TokenManager

        # Resolve API key (prefer ANTHROPIC_API_KEY, fallback to legacy CLAUDE_API_KEY)
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")

        if not api_key or api_key.strip() in ("your-claude-api-key-here", "your-anthropic-api-key-here", ""):
            print("[X] Anthropic API key not configured. Please set ANTHROPIC_API_KEY in your .env file.")
            print("    Get your API key and manage billing at:")
            print("    - https://platform.claude.com/ (dashboard)")
            legacy = os.getenv("CLAUDE_API_KEY")
            if legacy and legacy.strip():
                print("    Detected CLAUDE_API_KEY; prefer using ANTHROPIC_API_KEY going forward.")
            return

        # Ensure downstream clients can read the key from the standard env var
        os.environ["ANTHROPIC_API_KEY"] = api_key

        # Initialize TokenManager for tracking
        token_manager = TokenManager()

        # Recommended Claude model and configuration (allow env override)
        preferred_models = [
            os.getenv("ANTHROPIC_MODEL", "").strip(),
            "claude-haiku-4-5-20251001",    # widely available, cheaper, $1 / input MTok $5 / output MTok, as of 02/2026
        ]
        # pick first non-empty
        model_name = next((m for m in preferred_models if m), "claude-haiku-4-5-20251001")

        llm = ChatAnthropic(
            model=model_name,
            temperature=0.7
        )

        print("[OK] Anthropic Claude initialized successfully!")
        print(f"Using model: {model_name}")

        # Minimal prompt
        prompt = "Hi Claude!"
        # Estimate prompt tokens (approximate)
        approx_tokens = token_manager.count_tokens(prompt, model_name)
        print(f"\nApprox prompt tokens: {approx_tokens}")

        # Build messages
        messages = [HumanMessage(content=prompt)]

        # Call Claude
        print("\nSending request to Claude...\n")
        try:
            response = llm.generate([messages])
        except Exception as api_err:
            err_msg = str(api_err)
            print("[X] Claude API error:", err_msg)
            low = err_msg.lower()
            if "credit balance is too low" in low or "billing" in low:
                print("\n[Hint] Your Anthropic credit balance appears too low.")
                print("Billing dashboard:")
                print(" - https://platform.claude.com/")
                print(" - https://console.anthropic.com/")
            if "not_found_error" in low or "model:" in low or "404" in low:
                print("\n[Hint] The requested Claude model was not found or not enabled.")
                print(f"- Requested model: {model_name}")
                print("- Try: claude-3-5-sonnet-latest")
                print("- Or set ANTHROPIC_MODEL in .env to a model you have access to")
            return

        # Print concise response
        if response.generations and response.generations[0]:
            answer = response.generations[0][0].text.strip()
            print("Claude:", answer)

            # Track output tokens (approximate)
            output_tokens = token_manager.count_tokens(answer, model_name)
            print(f"Output tokens (approx): {output_tokens}")

            # Log usage
            token_manager.log_usage(model_name, approx_tokens, "claude_prompt", is_output=False)
            token_manager.log_usage(model_name, output_tokens, "claude_response", is_output=True)

    except ImportError as ie:
        print("[X] Required packages not installed or imports failed.")
        print("    Install/upgrade with: pip install -U langchain-anthropic anthropic")
        print("    If you see message class import errors, also ensure LangChain is up-to-date.")
    except Exception as e:
        print(f"[X] Error using Claude: {e}")
        
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

    # Try Claude example
    print("Testing Claude Integration:")
    example_claude_usage()
    print()
    
    # Try Hugging Face example
    print("Testing Hugging Face Integration:")
    example_huggingface_usage()
    print()
    
    # print("Remember to:")
    # print("1. Set your actual API keys in the .env file")
    # print("2. Never commit the .env file to version control")
    # print("3. Use different .env files for different environments")

if __name__ == "__main__":
    main()