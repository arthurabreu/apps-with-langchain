"""
Main application file for LangChain project.
This file demonstrates how to use environment variables with python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Token management
from token_utils import TokenManager

def run_sample_openai_call(token_manager: TokenManager, model: str = "gpt-4o-mini"):
    """
    Demonstrates making an OpenAI call and tracking token usage.
    Uses TokenManager to count prompt/response tokens and estimate costs.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("\n" + "=" * 40)
        print("[!] langchain-openai not installed. Skipping OpenAI call demo.")
        print("=" * 40 + "\n")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("\n" + "=" * 40)
        print("[!] OPENAI_API_KEY not configured. Skipping OpenAI call demo.")
        print("=" * 40 + "\n")
        return

    # Build LLM
    llm = ChatOpenAI(model=model, temperature=0.2, api_key=api_key)

    # Example prompt
    prompt = "Hello! How are you?"

    # Count prompt tokens and show remaining budget
    prompt_tokens = token_manager.count_tokens(prompt, model)
    tokens_used, remaining, usage_percent = token_manager.check_context_window(prompt_tokens, model)
    est_prompt_cost = token_manager.estimate_cost(prompt_tokens, model, is_output=False)

    print("\n[TokenManager] Prompt analysis")
    print(f"- Model: {model}")
    print(f"- Prompt tokens: {prompt_tokens}")
    print(f"- Context remaining before call: {remaining} tokens ({usage_percent:.3f}% used)")
    print(f"- Estimated prompt input cost: ${est_prompt_cost:.6f}")

    # Perform the call
    try:
        response = llm.invoke(prompt)
        print(f"Response: {response.content}")

        # Count response tokens (rough estimate; provider usage is authoritative)
        response_text = getattr(response, "content", str(response))
        response_tokens = token_manager.count_tokens(response_text, model)
        est_output_cost = token_manager.estimate_cost(response_tokens, model, is_output=True)

        print("\n[TokenManager] Response analysis")
        print(f"- Response tokens: {response_tokens}")
        print(f"- Estimated output cost: ${est_output_cost:.6f}")

        # Log usage
        token_manager.log_usage(model, prompt_tokens, "openai_prompt", is_output=False)
        token_manager.log_usage(model, response_tokens, "openai_completion", is_output=True)

        # Show updated summary
        summary = token_manager.get_usage_summary()
        print("\n[TokenManager] Session usage summary")
        print(f"- Total tokens: {summary['total_tokens']}")
        print(f"- Total estimated cost: ${summary['total_cost']:.6f}")

    except Exception as e:
        print("\n" + "=" * 40)
        print(f"[X] Error during OpenAI call:")
        print(f"{e}")
        print("=" * 40 + "\n")

def main():
    """
    Main function demonstrating environment variable usage and token tracking.
    """
    print("LangChain Environment Setup")
    print("=" * 40)
    
    # Initialize token manager
    token_manager = TokenManager()

    # Get API keys from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Check status of each key
    openai_configured = openai_key and openai_key != "your-openai-api-key-here"
    google_configured = google_key and google_key != "your-google-api-key-here"
    huggingface_configured = huggingface_key and huggingface_key != "your-huggingface-api-key-here"
    anthropic_configured = anthropic_key and anthropic_key != "your-anthropic-api-key-here"
    
    # Count how many keys are configured
    configured_count = sum([openai_configured, google_configured, huggingface_configured, anthropic_configured])
    
    # Only show API Key Status block if NO keys are configured
    if configured_count == 0:
        print("\nAPI Key Status:")
        print(f"OpenAI: [X] Not configured")
        print(f"Google: [X] Not configured")
        print(f"Hugging Face: [X] Not configured")
        print(f"Anthropic: [X] Not configured")
        
        print("\n" + "=" * 40)
        print("To configure your API keys:")
        print("1. Edit the .env file in the project root")
        print("2. Replace 'your-api-key-here' with your actual API keys")
        print("3. Never commit the .env file to version control!")
        print("=" * 40 + "\n")
    else:
        print("\nAPI Key Status:")
        print(f"OpenAI: {'[OK] Configured' if openai_configured else '[X] Not configured'}")
        print(f"Google: {'[OK] Configured' if google_configured else '[X] Not configured'}")
        print(f"Hugging Face: {'[OK] Configured' if huggingface_configured else '[X] Not configured'}")
        print(f"Anthropic: {'[OK] Configured' if anthropic_configured else '[X] Not configured'}")
    
    # Environment settings
    environment = os.getenv("ENVIRONMENT", "production")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"Environment: {environment}")
    print(f"Debug Mode: {debug}\n")

    # Demonstrate a sample OpenAI call with token tracking
    print("[DEMO] Running sample OpenAI call with TokenManager...")
    run_sample_openai_call(token_manager, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

if __name__ == "__main__":
    main()

    # --- EXAMPLES FROM example_langchain_usage.py ---
    try:
        from example_langchain_usage import main as example_langchain_main
        print("\n[DEMO] Now running LangChain Environment Variable Examples:\n")
        example_langchain_main()
    except ImportError:
        print("[!] Cannot find example_langchain_usage.py or its main() function.")
