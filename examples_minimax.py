"""
Example script demonstrating how to use MiniMax-M2.1 model in the project.
This shows the three main ways to integrate MiniMax:
1. Using the model factory to create a MiniMax model
2. Using the convenience method for default MiniMax setup
3. Using custom HuggingFace models with the same infrastructure
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.dependency_injection import get_container
from src.core.config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from src.core.interfaces import ModelConfig


def example_1_factory_method():
    """Example 1: Using the factory directly with ModelConfig"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Creating MiniMax model via Factory")
    print("=" * 60)
    
    container = get_container()
    factory = container.resolve('model_factory')
    
    # Create a MiniMax model with custom configuration
    config = ModelConfig(
        model_name="MiniMaxAI/MiniMax-M2.1",
        temperature=1.0,  # MiniMax docs recommend 1.0
        max_tokens=512
    )
    
    try:
        model = factory.create_model("minimax", config)
        print(f"✓ Created MiniMax model: {model.get_model_info()}")
        
        # Generate text
        result = model.generate(
            "Explain quantum computing in simple terms.",
            skip_prompt=True
        )
        print(f"\nGenerated response:\n{result.content[:200]}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_2_convenience_method():
    """Example 2: Using the convenience method for MiniMax"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Creating MiniMax model via Convenience Method")
    print("=" * 60)
    
    container = get_container()
    factory = container.resolve('model_factory')
    
    # Create MiniMax model with default settings
    try:
        model = factory.create_minimax_model(
            temperature=1.0,
            max_tokens=512
        )
        print(f"✓ Created MiniMax model with defaults")
        print(f"  Model info: {model.get_model_info()}")
        
        # Generate text
        result = model.generate(
            "What is machine learning?",
            skip_prompt=True
        )
        print(f"\nGenerated response:\n{result.content[:200]}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_3_custom_huggingface():
    """Example 3: Using any HuggingFace model via the factory"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Creating Custom HuggingFace Model")
    print("=" * 60)
    
    container = get_container()
    factory = container.resolve('model_factory')
    
    # You can use ANY HuggingFace model (not just MiniMax)
    custom_model_id = "mistralai/Mistral-7B-v0.1"  # Example: another open-source model
    
    try:
        model = factory.create_huggingface_model(
            model_name=custom_model_id,
            temperature=0.7,
            max_tokens=256
        )
        print(f"✓ Created HuggingFace model: {custom_model_id}")
        print(f"  Model info: {model.get_model_info()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_interchangeable_models():
    """Example 4: Show how Claude and MiniMax are interchangeable"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Interchangeable Model Usage")
    print("=" * 60)
    
    container = get_container()
    factory = container.resolve('model_factory')
    
    print("Both Claude and MiniMax models implement ILanguageModel")
    print("So you can switch between them seamlessly:\n")
    
    try:
        # Create Claude model
        claude = factory.create_default_claude_model()
        print(f"✓ Claude model: {claude.get_model_info()['provider']}")
        
        # Create MiniMax model
        minimax = factory.create_minimax_model()
        print(f"✓ MiniMax model: {minimax.get_model_info()['provider']}")
        
        # Both implement the same interface
        print(f"\nBoth have the same methods:")
        print(f"  - generate(prompt) -> GenerationResult")
        print(f"  - get_model_info() -> Dict[str, Any]")
        print(f"  - provider property")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_5_comparing_providers():
    """Example 5: Comparing available providers"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Available Providers")
    print("=" * 60)
    
    container = get_container()
    factory = container.resolve('model_factory')
    
    providers = factory.get_available_providers()
    print(f"Available providers: {', '.join(providers)}")
    print("\nProviders:")
    print("  - 'anthropic': Uses Anthropic's Claude API (requires API key)")
    print("  - 'minimax': Uses MiniMax-M2.1 (local, no API key needed)")
    print("  - 'huggingface': Uses any HuggingFace model (local, no API key needed)")


# Configuration guide
SETUP_GUIDE = """
===============================================================================
SETUP GUIDE: Using MiniMax-M2.1 in Your Project
===============================================================================

1. INSTALLATION
   Make sure you have the required dependencies:
   ```bash
   pip install transformers torch
   ```
   
   For faster inference (optional):
   ```bash
   pip install accelerate  # For device_map="auto"
   ```

2. HuggingFace TOKEN (Optional, for private models)
   If using private models or to avoid rate limiting:
   ```bash
   # Create a .env file in your project root with:
   HUGGINGFACE_API_KEY=your_token_here
   ```
   
   Get your token from: https://huggingface.co/settings/tokens

3. MEMORY REQUIREMENTS
   MiniMax-M2.1 is 229B parameters:
   - Full precision: ~450GB VRAM (requires multiple GPUs)
   - Recommended: Use quantized versions (~60GB VRAM with bfloat16)
   
   Check quantized versions:
   https://huggingface.co/models?other=base_model:quantized:MiniMaxAI/MiniMax-M2.1

4. BASIC USAGE

   # Method A: Using the factory
   from src.core.dependency_injection import get_container
   from src.core.interfaces import ModelConfig
   
   container = get_container()
   factory = container.resolve('model_factory')
   
   config = ModelConfig(
       model_name="MiniMaxAI/MiniMax-M2.1",
       temperature=1.0,
       max_tokens=512
   )
   model = factory.create_model("minimax", config)
   
   # Method B: Using convenience method
   model = factory.create_minimax_model()
   
   # Method C: Using any HuggingFace model
   model = factory.create_huggingface_model("mistralai/Mistral-7B-v0.1")
   
   # Generate text
   result = model.generate("Your prompt here")
   print(result.content)

5. RECOMMENDED PARAMETERS (per MiniMax docs)
   - temperature: 1.0 (for optimal results)
   - top_p: 0.95
   - top_k: 40

6. SWITCHING FROM CLAUDE TO MINIMAX
   
   # Your code can be the same - both implement ILanguageModel:
   def my_function(model: ILanguageModel):
       result = model.generate("prompt")
       return result.content
   
   # Use with Claude:
   claude = factory.create_default_claude_model()
   my_function(claude)
   
   # Use with MiniMax:
   minimax = factory.create_minimax_model()
   my_function(minimax)

7. COST & PRICING
   - Claude: API costs (see Anthropic pricing)
   - MiniMax local: FREE (runs on your hardware)
   - Standard HuggingFace models: FREE (runs on your hardware)

===============================================================================
"""

if __name__ == "__main__":
    # Print setup guide
    print(SETUP_GUIDE)
    
    # Run examples
    try:
        example_5_comparing_providers()
        # Uncomment to run other examples (requires the model to be downloaded)
        # example_1_factory_method()
        # example_2_convenience_method()
        # example_4_interchangeable_models()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
