"""
Main application file for LangChain project.
Focused on testing local Hugging Face models with LangChain.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api.deprecation")

import os
import sys
# Add project root and src to path so we can import files correctly before importing from core
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(1, src_dir)

from dotenv import load_dotenv
from core.config import (
    DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    INTERACTIVE_MAX_TOKENS, RESPONSES_DIR
)

# Load environment variables from .env file
load_dotenv()
print(f"Debug: Key found? {os.getenv('ANTHROPIC_API_KEY') is not None}")

# Set HF_TOKEN from HUGGINGFACE_API_KEY for HuggingFace library compatibility
hf_key = os.getenv("HUGGINGFACE_API_KEY")
if hf_key:
    os.environ["HF_TOKEN"] = hf_key

def print_api_key_status():
    """Display the status of API keys."""
    keys = {
        "Hugging Face": os.getenv("HUGGINGFACE_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY")
    }
    
    print("\n" + "=" * 40)
    print("API Key Status:")
    print("-" * 40)
    for name, key in keys.items():
        is_ok = key and "your-" not in key.lower() and key.strip() != ""
        print(f"{name:13}: {'[OK] Configured' if is_ok else '[X] Missing'}")
    print("=" * 40 + "\n")

def check_memory_usage():
    """Check current memory usage."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        print("\n" + "=" * 40)
        print("Memory Usage Report")
        print("=" * 40)
        print(f"RSS (Resident Set Size): {mem_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS (Virtual Memory Size): {mem_info.vms / 1024 / 1024:.2f} MB")
        print(f"Percent of system RAM: {process.memory_percent():.1f}%")
        
        # Get system-wide memory info
        system_memory = psutil.virtual_memory()
        print(f"\nSystem Memory:")
        print(f"  Total: {system_memory.total / 1024 / 1024:.0f} MB")
        print(f"  Available: {system_memory.available / 1024 / 1024:.0f} MB")
        print(f"  Used: {system_memory.used / 1024 / 1024:.0f} MB")
        print(f"  Percent: {system_memory.percent}%")
        
        # Check GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\nGPU Memory:")
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(i) / 1024 / 1024
                    cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                    print(f"  GPU {i}: Allocated: {alloc:.2f} MB, Cached: {cached:.2f} MB")
        except ImportError:
            print("\n[INFO] PyTorch not available for GPU memory check")
        
        # Check if memory seems high
        if process.memory_percent() > 30:
            print("\n⚠️  WARNING: High memory usage detected!")
            print("   Consider cleaning up model resources if you're done testing.")
            print("   You can use the 'Clean up model memory' option in the menu.")
        
        print("=" * 40)
        
    except ImportError:
        print("\n[INFO] psutil not installed. Install with: pip install psutil")
        print("[INFO] Memory check requires: pip install psutil")
    except Exception as e:
        print(f"[ERROR] Could not check memory: {e}")

def prompt_continue_or_skip():
    """
    Prompt user to continue to next test or skip remaining tests.
    
    Returns:
        bool: True to continue, False to skip remaining tests
    """
    while True:
        choice = input("\nPress Enter to continue to next test, or 's' to skip remaining tests: ").strip().lower()
        if choice == "":
            return True  # Continue to next test
        elif choice == "s":
            return False  # Skip remaining tests
        else:
            print("Invalid input. Press Enter to continue or 's' to skip.")

def cleanup_model_memory():
    """Clean up model memory resources."""
    print("\n" + "=" * 40)
    print("Cleaning Up Model Memory")
    print("=" * 40)
    
    print("\n[INFO] This will attempt to free up memory used by loaded models.")
    
    try:
        import gc
        import torch
        
        # Get initial memory
        initial_memory = None
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
        except:
            pass
        
        print("\n[STEP 1] Running Python garbage collection...")
        gc.collect()
        print("[SUCCESS] Garbage collection completed")
        
        print("\n[STEP 2] Clearing PyTorch CUDA cache (if using GPU)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("[SUCCESS] CUDA cache cleared")
        else:
            print("[INFO] No GPU detected, skipping CUDA cache clear")
        
        print("\n[STEP 3] Clearing other caches...")
        # Clear any remaining model references
        if 'local_model' in globals():
            try:
                local_model.cleanup()
                del local_model
                print("[SUCCESS] Local model cleaned up")
            except:
                pass
        
        # Force another GC
        gc.collect()
        
        # Check memory after cleanup
        if initial_memory:
            try:
                final_memory = process.memory_info().rss
                freed = (initial_memory - final_memory) / 1024 / 1024
                if freed > 0:
                    print(f"\n[SUCCESS] Freed approximately {freed:.2f} MB of memory")
                else:
                    print(f"\n[INFO] Memory usage remained about the same")
            except:
                pass
        
        print("\n" + "=" * 40)
        print("[SUCCESS] Cleanup completed!")
        print("[INFO] You can now test models again with fresh memory.")
        print("=" * 40)
        
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")

def check_model_cached_locally(model_id: str) -> bool:
    """
    Check if a HuggingFace model is cached locally.

    Args:
        model_id: HuggingFace model ID (e.g., "MiniMaxAI/MiniMax-M2.1")

    Returns:
        True if model is cached, False otherwise
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import model_info

        # Check if we can find the model locally via huggingface_hub
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            # Check if model directory exists in cache
            model_name_escaped = model_id.replace("/", "--")
            model_cache_path = os.path.join(cache_dir, f"models--{model_name_escaped}")
            return os.path.exists(model_cache_path)
        return False
    except:
        return False


def download_hf_model(model_id: str) -> bool:
    """
    Download a HuggingFace model if not already cached.

    Args:
        model_id: HuggingFace model ID

    Returns:
        True if model is available (downloaded or cached), False otherwise
    """
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download

        hf_token = os.getenv("HUGGINGFACE_API_KEY")

        print(f"\n[INFO] Downloading tokenizer for {model_id}...")
        AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token
        )
        print(f"[SUCCESS] Tokenizer downloaded")

        print(f"\n[INFO] Downloading model {model_id} (this may take several minutes)...")
        print("[INFO] Model size: Check HuggingFace page for exact size")

        # Use snapshot_download to download files without loading into memory
        snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_dir_use_symlinks=False
        )

        print(f"[SUCCESS] Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        return False


def select_huggingface_model() -> str:
    """
    Let user select which HuggingFace model to test from local folder.

    Returns:
        Model ID string or None if user cancels
    """
    from core.hf_model_manager import get_hf_models_folder, get_model_folder_size
    from pathlib import Path

    print("\n" + "=" * 60)
    print("Select HuggingFace Model to Test")
    print("=" * 60)

    # Get local models from folder
    hf_models_folder = get_hf_models_folder()
    local_models = []

    if hf_models_folder.exists():
        for item in sorted(hf_models_folder.iterdir()):
            if item.is_dir() and any(item.iterdir()) and not item.name.startswith('.'):
                local_models.append(item)

    if not local_models:
        print("[INFO] No local models found in folder.")
        return None

    print("\nLocal models found:")
    for idx, folder in enumerate(local_models, 1):
        size = get_model_folder_size(folder)
        print(f"{idx}. {folder.name} ({size})")

    print(f"{len(local_models) + 1}. Enter custom HuggingFace model ID")
    print()

    choice = input(f"Select model [1-{len(local_models) + 1}]: ").strip()

    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(local_models):
            return local_models[choice_idx].name
        elif choice_idx == len(local_models):
            # Custom model
            model_id = input("\nEnter HuggingFace model ID (e.g., 'username/model-name'): ").strip()
            return model_id if model_id else None
        else:
            print("[ERROR] Invalid choice.")
            return None
    except ValueError:
        print("[ERROR] Please enter a number.")
        return None


def test_local_model():
    """Test a selected HuggingFace model with different prompts."""
    print("\n[TEST] Local HuggingFace Model Testing")
    print("=" * 60)

    try:
        # Let user select model
        model_id = select_huggingface_model()
        if not model_id:
            return

        # Check memory before loading
        print("\n[INFO] Checking system memory...")
        check_memory_usage()

        # Check if model is cached
        is_cached = check_model_cached_locally(model_id)
        print(f"\n[INFO] Model cache status: {'✓ Cached locally' if is_cached else '✗ Not cached'}")

        if not is_cached:
            # Ask user to download
            download_choice = input(f"\nModel '{model_id}' is not cached locally.\nDownload now? (y/n): ").lower()
            if download_choice != 'y':
                print("[INFO] Model download skipped. Exiting.")
                return

            # Download the model
            if not download_hf_model(model_id):
                print("[ERROR] Failed to download model. Cannot proceed.")
                return

        # Create and test the model using factory
        print("\n" + "=" * 60)
        print("Initializing Model for Testing")
        print("=" * 60)

        from core.dependency_injection import get_container
        from core.interfaces import ModelConfig

        container = get_container()
        factory = container.get_model_factory()

        # Create model config
        config = ModelConfig(
            model_name=model_id,
            temperature=1.0,
            max_tokens=512
        )

        print(f"\n[INFO] Creating model instance: {model_id}")
        model = factory.create_model("huggingface", config)

        print("[SUCCESS] Model loaded and ready!")

        # Test prompts
        test_prompts = [
            {
                "name": "Kotlin Coroutine",
                "prompt": "Write a Kotlin function that uses coroutines to fetch data from two APIs concurrently. Include error handling and timeouts."
            },
            {
                "name": "Kotlin Palindrome",
                "prompt": "Write a Kotlin function that checks if a string is a palindrome. Make it case-insensitive and ignore non-alphanumeric characters."
            },
            {
                "name": "Explain Coroutines",
                "prompt": "Explain Kotlin coroutines to a beginner. Keep it under 150 words."
            }
        ]

        for i, test in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Testing: {test['name']}")
            print(f"Prompt: {test['prompt']}")
            print("-" * 60)

            try:
                result = model.generate(test['prompt'], skip_prompt=True)
                print(f"Response:\n{result.content}")
                print(f"\n[INFO] Tokens used: {result.tokens_used}")
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")

            if i < len(test_prompts):
                if not prompt_continue_or_skip():
                    print("\n[INFO] Skipping remaining tests...")
                    break

        print("\n" + "=" * 60)
        print("Model Testing Complete!")
        print("=" * 60)

        # Ask if user wants to clean up
        cleanup_choice = input("\nClean up model memory? (y/n): ").lower()
        if cleanup_choice == 'y':
            print("\n[INFO] Cleaning up model resources...")
            model.cleanup()
            del model
            import gc
            gc.collect()
            print("[SUCCESS] Model memory cleaned up!")
            check_memory_usage()
        else:
            print("[INFO] Model remains loaded in memory.")
            print("[INFO] You can clean up later from the main menu.")

        print("=" * 60)

    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("Install with: pip install transformers torch accelerate")
    except Exception as e:
        print(f"[ERROR] Failed to test model: {e}")
        import traceback
        traceback.print_exc()

def test_claude_model():
    """Test the Claude model with different prompts."""
    print("[TEST] Claude Model Demo")
    print("-" * 40)

    # Check if Claude API key is configured
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_key or "your-" in claude_key.lower():
        print("\n[ERROR] Anthropic API key not configured.")
        print("Add your ANTHROPIC_API_KEY to the .env file")
        print("Format: ANTHROPIC_API_KEY=your-actual-key-here")
        return

    try:
        from core.dependency_injection import get_container
        from core.interfaces import ModelConfig

        # Get services from DI container
        container = get_container()
        factory = container.get_model_factory()

        # Test prompts
        test_prompts = [
            {
                "name": "Kotlin Coroutine",
                "prompt": "Write a Kotlin function that uses coroutines to fetch data from two APIs concurrently. Include error handling and timeouts."
            },
            {
                "name": "Kotlin Palindrome",
                "prompt": "Write a Kotlin function that checks if a string is a palindrome. Make it case-insensitive and ignore non-alphanumeric characters."
            },
            {
                "name": "Explain Coroutines",
                "prompt": "Explain Kotlin coroutines to a beginner. Keep it under 150 words."
            }
        ]

        for i, test in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Testing: {test['name']}")
            print(f"Prompt: {test['prompt']}")
            print("-" * 40)

            try:
                config = ModelConfig(
                    model_name="claude-3-haiku-20240307",
                    system_message="You are a helpful assistant.",
                    max_tokens=512,
                    temperature=0.2,
                )
                claude_model = factory.create_model("anthropic", config)
                result = claude_model.generate(test['prompt'])
                print(f"Response:\n{result.content}")
            except Exception as e:
                print(f"Error: {e}")

            if i < len(test_prompts):
                input("\nPress Enter to continue to next test...")

        print("\n" + "=" * 40)
        print("Claude Model Testing Complete!")
        print("=" * 40)

    except ImportError as e:
        print(f"[ERROR] Missing dependencies for Claude: {e}")
        print("Install with: pip install langchain-anthropic")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Claude model: {e}")

def compare_models():
    """Compare all available models."""
    print("\n" + "=" * 60)
    print("Model Comparison Tool")
    print("=" * 60)
    
    try:
        from core.model_comparison import ModelComparison
        
        # Check which models are available based on API keys
        available_models = []
        
        # Check local model
        try:
            from core.langchain_huggingface_local import LocalHuggingFaceModel
            available_models.append("Local Hugging Face")
        except:
            print("[INFO] Local model not available (missing dependencies)")
        
        # Check OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and "your-" not in openai_key.lower():
            try:
                from core.openai_model import OpenAIModel
                available_models.append("OpenAI GPT")
            except:
                print("[INFO] OpenAI model not available (missing dependencies)")
        else:
            print("[INFO] OpenAI API key not configured")
        
        # Check Claude
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_key and "your-" not in claude_key.lower():
            try:
                from core.claude_model import ClaudeModel
                available_models.append("Claude")
            except:
                print("[INFO] Claude model not available (missing dependencies)")
        else:
            print("[INFO] Claude API key not configured")
        
        if len(available_models) < 2:
            print(f"\n[WARNING] Only {len(available_models)} model(s) available.")
            print("You need at least 2 models for comparison.")
            print("\nAvailable models:")
            for model in available_models:
                print(f"  - {model}")
            
            if len(available_models) == 1:
                choice = input(f"\nWould you like to test the {available_models[0]} model instead? (y/n): ").lower()
                if choice == 'y':
                    if "Local" in available_models[0]:
                        test_local_model()
                    elif "OpenAI" in available_models[0]:
                        test_openai_model()
                    elif "Claude" in available_models[0]:
                        test_claude_model()
            return
        
        print(f"\n[INFO] Comparing {len(available_models)} models:")
        for model in available_models:
            print(f"  - {model}")
        
        # Define test prompts
        test_prompts = [
            {
                "name": "Kotlin Coroutine",
                "prompt": "Write a Kotlin function that uses coroutines to fetch data from two APIs concurrently. Include error handling and timeouts."
            },
            {
                "name": "Kotlin Palindrome", 
                "prompt": "Write a Kotlin function that checks if a string is a palindrome. Make it case-insensitive and ignore non-alphanumeric characters."
            }
        ]
        
        # Ask user if they want to customize prompts
        print("\n[1] Use default test prompts")
        print("[2] Enter custom prompts")
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "2":
            test_prompts = []
            print("\nEnter test prompts (enter 'done' when finished):")
            i = 1
            while True:
                name = input(f"\nPrompt {i} name: ").strip()
                if name.lower() == 'done':
                    break
                prompt = input(f"Prompt {i} text: ").strip()
                test_prompts.append({"name": name, "prompt": prompt})
                i += 1
            
            if not test_prompts:
                print("[INFO] No custom prompts entered, using defaults.")
                test_prompts = [
                    {
                        "name": "Kotlin Coroutine",
                        "prompt": "Write a Kotlin function that uses coroutines to fetch data from two APIs concurrently."
                    },
                    {
                        "name": "Kotlin Palindrome",
                        "prompt": "Write a Kotlin palindrome checker function."
                    }
                ]
        
        # Run comparison
        comparison = ModelComparison()
        comparison.run_comparison(test_prompts)
        
        # Ask if user wants to save results
        save_choice = input("\nWould you like to save the comparison results? (y/n): ").lower()
        if save_choice == 'y':
            from core.services import get_brazil_time
            import json

            brazil_time = get_brazil_time()
            timestamp = brazil_time.strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{timestamp}.json"

            # Prepare results
            results = {
                "timestamp": brazil_time.isoformat(),
                "models_compared": available_models,
                "prompts": test_prompts,
                "results": comparison.results
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"[SUCCESS] Results saved to {filename}")
        
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("Install required packages: pip install langchain-openai langchain-anthropic")
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")

def explain_langchain_concepts():
    """Explain key LangChain concepts relevant to all models."""
    concepts = {
        "LLM Wrappers": """
LangChain provides wrappers for different LLM providers:
- ChatOpenAI: For OpenAI models (GPT-3.5, GPT-4, etc.)
- ChatAnthropic: For Claude models (Claude-3, etc.)
- HuggingFacePipeline: For local Hugging Face models
- HuggingFaceEndpoint: For Hugging Face Inference API

These wrappers provide a consistent interface so you can switch
between models without changing your application logic.
""",
        "Prompt Templates": """
Prompt templates help structure your prompts consistently:
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {language} expert."),
    ("user", "{question}")
])

prompt = template.format_messages(
    language="Kotlin",
    question="Write a palindrome function"
)

You can use the same template with different models!
""",
        "Chains": """
Chains combine multiple components:
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Create a chain: prompt -> model -> output parser
chain = prompt | llm | StrOutputParser()

# Use with any model
result = chain.invoke({"question": "Your prompt here"})

Chains make it easy to swap models without changing your logic.
""",
        "Model Types Comparison": """
Different model types have different strengths:

LOCAL MODELS (Hugging Face):
✓ No API costs
✓ Complete privacy
✓ Customizable/Finetunable
✓ Offline capability
✗ Requires GPU for good performance
✗ Limited context window

CLOUD MODELS (OpenAI/Claude):
✓ Easy to use
✓ High performance
✓ Large context windows
✓ Regular updates
✗ API costs
✗ Privacy concerns
✗ Requires internet connection
""",
        "Memory Management": """
Important memory considerations for local models:

1. Large models consume significant RAM (4B+ parameters)
2. GPU memory needed for faster inference
3. Always clean up after testing:
   - Use context manager (with statement)
   - Call model.cleanup() when done
   - Restart Python kernel if needed
4. Close VS Code to free all memory
5. Monitor with check_memory_usage()
""",
        "Choosing the Right Model": """
Consider these factors when choosing a model:

1. Privacy requirements: Use local models for sensitive data
2. Budget: Local models are free, cloud models have costs
3. Performance needs: Cloud models often perform better
4. Internet access: Local models work offline
5. Customization: Local models can be fine-tuned
6. Hardware: Local models need sufficient RAM/GPU

In this lab, you can test all options and decide what works best!
"""
    }
    
    print("\n" + "=" * 40)
    print("LangChain Learning Guide")
    print("=" * 40)
    print("Understanding different model types and when to use them.")
    
    for i, (title, content) in enumerate(concepts.items(), 1):
        print(f"\n{i}. {title}")
        print("-" * 30)
        print(content)
        
        if i < len(concepts):
            input("\nPress Enter for next concept...")

def show_examples():
    """Show example usage of different models."""
    print("\n" + "=" * 60)
    print("Example Code Snippets")
    print("=" * 60)
    
    examples = {
        "Local Model": """
from core.langchain_huggingface_local import LocalHuggingFaceModel

# Initialize with custom parameters
model = LocalHuggingFaceModel(
    model_id="JetBrains/Mellum-4b-sft-kotlin",
    max_length=256,
    temperature=0.3
)

# Generate code
response = model.generate(
    "Write a Kotlin function to sort a list"
)

# Clean up when done
model.cleanup()
""",
        "OpenAI Model": """
from core.openai_model import OpenAIModel

# Initialize with GPT-4
model = OpenAIModel(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1024
)

# Generate with custom prompt
response = model.generate(
    "Explain Kotlin coroutines with examples"
)
""",
        "Claude Model": """
from core.claude_model import ClaudeModel

# Initialize Claude
model = ClaudeModel(
    model_name="claude-3-opus-20240229",
    temperature=0.5,
    max_tokens=500
)

# Get structured response
response = model.generate(
    "Write a palindrome function with documentation"
)
""",
        "Memory Management": """
# Check current memory usage
check_memory_usage()

# Test model with automatic cleanup
from core.langchain_huggingface_local import LocalHuggingFaceModel

with LocalHuggingFaceModel() as model:
    response = model.generate("Hello")
    print(response)

# Memory is automatically cleaned up!

# Or manually clean up
model2 = LocalHuggingFaceModel()
# ... use model ...
model2.cleanup()
del model2

# Force garbage collection
import gc
gc.collect()
""",
        "Model Comparison": """
from core.model_comparison import ModelComparison

# Create comparison
comparison = ModelComparison()

# Test with custom prompts
prompts = [
    {"name": "Task 1", "prompt": "Write hello world in Kotlin"},
    {"name": "Task 2", "prompt": "Explain coroutines"}
]

# Run comparison
comparison.run_comparison(prompts)

# View results
comparison.print_summary()
"""
    }
    
    for title, code in examples.items():
        print(f"\n{title}:")
        print("-" * 40)
        print(code)
        input("\nPress Enter for next example...")
    
    print("\n[INFO] These examples are available in the example_langchain_usage.py file")

def show_system_info():
    """Show system information and requirements."""
    print("\n" + "=" * 60)
    print("System Information & Requirements")
    print("=" * 60)
    
    import platform
    
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\nMemory:")
        print(f"  Total: {memory.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Available: {memory.available / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Used: {memory.percent}%")
        
        # Recommendations
        print(f"\nRequirements for local models:")
        print(f"  Minimum: 8GB RAM (for small models)")
        print(f"  Recommended: 16GB+ RAM (for 4B+ parameter models)")
        print(f"  GPU: Optional but recommended for speed")
        
        if memory.total < 8 * 1024 * 1024 * 1024:  # Less than 8GB
            print(f"\n⚠️  WARNING: System has less than 8GB RAM.")
            print("   Local models may not work well.")
            print("   Consider using cloud models (Claude) instead.")
    except:
        pass
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"\nDisk Space:")
        print(f"  Total: {total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Free: {free / 1024 / 1024 / 1024:.1f} GB")
        
        # Models can be large (4B model ~8GB)
        if free < 10 * 1024 * 1024 * 1024:  # Less than 10GB free
            print(f"\n⚠️  WARNING: Less than 10GB disk space free.")
            print("   Model downloads may fail or fill your disk.")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("Tips:")
    print("  • Close other applications when testing local models")
    print("  • Use check_memory_usage() to monitor RAM")
    print("  • Clean up model memory when done testing")
    print("  • Restart VS Code if memory gets too high")
    print("=" * 60)

def select_context():
    """
    Display context menu and return selected prompt context details.

    Returns:
        Tuple of (context_key, system_message, max_tokens, temperature)
    """
    from core.dependency_injection import get_container

    container = get_container()
    prompt_manager = container.get_prompt_manager()
    prompts = prompt_manager.list_prompts()

    keys = list(prompts.keys())

    print("\n" + "=" * 60)
    print("Select Development Context")
    print("=" * 60)

    for i, (key, desc) in enumerate(prompts.items(), 1):
        print(f"{i}. {desc}")

    print()
    choice = input("Enter choice (number): ").strip()

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(keys):
            print("[ERROR] Invalid choice.")
            return None

        selected_key = keys[idx]
        overrides = prompt_manager.get_config_overrides(selected_key)

        return (
            selected_key,
            prompt_manager.get_system_message(selected_key),
            overrides.get("max_tokens", 512),
            overrides.get("temperature", 0.2),
        )
    except (ValueError, KeyError) as e:
        print(f"[ERROR] Invalid choice: {e}")
        return None


def run_test_no_context():
    """Run direct API test (no system context) - cost benchmark."""
    print("\n" + "=" * 70)
    print("Running Test: Direct API (No Context)")
    print("=" * 70)
    try:
        from test_claude_direct import test_direct_api
        test_direct_api()
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")


def run_test_python_context():
    """Run API test with Python Expert context - cost benchmark."""
    print("\n" + "=" * 70)
    print("Running Test: API with Python Expert Context")
    print("=" * 70)
    try:
        from test_claude_with_context import test_with_python_context
        test_with_python_context()
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")


def ask_with_context():
    """
    Interactive mode: select a context and ask a question to Claude.
    """
    from core.dependency_injection import get_container
    from core.interfaces import ModelConfig
    from core.services import get_brazil_time
    from pathlib import Path

    context_info = select_context()
    if context_info is None:
        return

    context_key, system_message, max_tokens, temperature = context_info

    print(f"\n[INFO] Using context: {context_key.upper()}")
    print("=" * 60)

    # Get user question
    question = input("\nEnter your question: ").strip()
    if not question:
        print("[ERROR] No question provided.")
        return

    # Check if Claude API key is configured
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_key or "your-" in claude_key.lower():
        print("\n[ERROR] Anthropic API key not configured.")
        print("Add your ANTHROPIC_API_KEY to the .env file")
        return

    try:
        # Get services from DI container
        container = get_container()
        factory = container.get_model_factory()

        # Create model with context
        config = ModelConfig(
            model_name=DEFAULT_MODEL,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print("\n[INFO] Generating response...")
        print("-" * 60)

        claude_model = factory.create_model("anthropic", config)
        result = claude_model.generate(question, context_key=context_key)

        print(f"\nResponse:\n{result.content}")
        print("\n" + "=" * 60)

        # Save response to file
        responses_dir = Path(RESPONSES_DIR)
        responses_dir.mkdir(exist_ok=True)

        brazil_time = get_brazil_time()
        timestamp = brazil_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{context_key}_{timestamp}.md"
        filepath = responses_dir / filename

        with open(filepath, "w") as f:
            f.write(f"# Question: {question}\n\n")
            f.write(f"**Context:** {context_key}\n\n")
            f.write(f"**Date:** {brazil_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Response\n\n{result.content}\n\n")
            f.write(f"## Metadata\n\n")
            f.write(f"- Input tokens: {result.metadata.get('prompt_tokens', 'N/A')}\n")
            f.write(f"- Output tokens: {result.metadata.get('response_tokens', 'N/A')}\n")
            f.write(f"- Total cost: ${result.cost:.6f}\n")

        print(f"[INFO] Response saved to: {filepath}")

    except Exception as e:
        print(f"[ERROR] Failed to generate response: {e}")


def prompt_for_task(agent_name: str = "Agent") -> str:
    """
    Prompt user to choose between loading a task from file or typing manually.

    Args:
        agent_name: Name of the agent for display purposes (e.g., "Android Agent", "Excel Agent")

    Returns:
        The task/prompt string
    """
    from pathlib import Path

    files_dir = Path(__file__).parent.parent / "files"

    # Get list of available script files
    script_files = []
    if files_dir.exists():
        script_files = sorted([f for f in files_dir.glob("*.txt")])

    print("\n" + "=" * 60)
    print(f"✍️  {agent_name} - Task Selection")
    print("=" * 60)

    if script_files:
        print("\nChoose how to provide the task/prompt:")
        print("1. Load from a script file")
        print("2. Type instructions manually\n")

        choice = input("Select [1 or 2]: ").strip()

        if choice == "1":
            print("\n📁 Available Scripts:")
            for i, script_file in enumerate(script_files, 1):
                print(f"  {i}. {script_file.name}")
            print(f"  {len(script_files) + 1}. Cancel\n")

            while True:
                file_choice = input("Select script [number]: ").strip()
                if file_choice.isdigit():
                    file_idx = int(file_choice) - 1
                    if 0 <= file_idx < len(script_files):
                        try:
                            with open(script_files[file_idx], "r", encoding="utf-8") as f:
                                task = f.read().strip()
                            print(f"\n✓ Loaded: {script_files[file_idx].name}\n")
                            return task
                        except Exception as e:
                            print(f"[ERROR] Failed to load file: {e}\n")
                            break
                    elif file_idx == len(script_files):
                        print("Cancelled.\n")
                        return prompt_for_task(agent_name)
                    else:
                        print("Invalid selection. Try again.\n")
                else:
                    print("Invalid input. Try again.\n")
        elif choice == "2":
            print("\n📝 Task Description:")
            task = input("Enter what you would like to generate: ").strip()
            if not task:
                print("[ERROR] Task cannot be empty.")
                return prompt_for_task(agent_name)
            return task
        else:
            print("[ERROR] Invalid choice. Please select 1 or 2.")
            return prompt_for_task(agent_name)
    else:
        # No script files available, just ask for input
        print("\n📝 Task Description:")
        task = input("Enter what you would like to generate: ").strip()
        if not task:
            print("[ERROR] Task cannot be empty.")
            return prompt_for_task(agent_name)
        return task


def run_android_agent():
    """Launch the Android code-gen agent."""
    try:
        from android_agent.cli import main as android_main

        # Get task from user (file or manual input)
        task = prompt_for_task("Android Code-Gen Agent")
        if not task:
            return

        # Pass task to the android agent
        android_main(task=task)
    except Exception as e:
        print(f"[ERROR] Failed to launch Android agent: {e}")


def run_hf_to_excel_agent():
    """
    Launch the LLM to Excel Agent using local Hugging Face models.
    Generates structured response and exports to Excel.
    """
    from core.dependency_injection import get_container
    from core.interfaces import ModelConfig
    from core.hf_model_manager import select_hf_model
    from core.services import get_brazil_time
    from pathlib import Path
    import json

    print("\n" + "=" * 60)
    print("🤖 LLM TO EXCEL AGENT (Local Hugging Face)")
    print("=" * 60)

    try:
        # 1. Select HF model
        result = select_hf_model()
        if result is None:
            return

        model_id, model_folder = result
        print(f"✓ Using model: {model_id}\n")

        # 2. Get task/prompt (from file or manual input)
        task = prompt_for_task("LLM to Excel Agent")
        if not task:
            return

        print("💡 Tip: If you use Markdown Tables or numbered sections, the Excel export will be more structured.\n")

        # 3. Choose output format for the LLM response
        print("\n📄 Select Response Format:")
        print("1. JSON (Recommended - Best for table structure in Excel) [DEFAULT]")
        print("2. Plain Text / Markdown")
        format_choice = input("Select [1 or 2, default=1]: ").strip()
        
        is_json = format_choice != "2"
        if is_json:
            task += "\n\nIMPORTANT: Return ONLY a valid JSON object or a list of objects. No other text."
            file_ext = ".json"
        else:
            file_ext = ".md"

        # 4. Initialize and run model
        container = get_container()
        factory = container.get_model_factory()
        
        config = ModelConfig(
            model_name=str(model_folder),
            temperature=0.7,
            max_tokens=1024
        )

        print("\n[INFO] Initializing model...")
        model = factory.create_model("huggingface", config)
        
        print("\n[INFO] Generating response...")
        gen_result = model.generate(task, skip_prompt=True)
        
        content = gen_result.content.strip()
        print(f"\nResponse received ({len(content)} characters).")

        # 5. Save response to file
        responses_dir = Path(RESPONSES_DIR) / "excel_exports"
        responses_dir.mkdir(parents=True, exist_ok=True)

        brazil_time = get_brazil_time()
        timestamp = brazil_time.strftime("%Y%m%d_%H%M%S")
        input_filename = f"llm_output_{timestamp}{file_ext}"
        input_filepath = responses_dir / input_filename

        with open(input_filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✓ Response saved to: {input_filepath}")

        # 6. Export to Excel
        output_filename = f"llm_report_{timestamp}.xlsx"
        output_filepath = responses_dir / output_filename
        
        print(f"\n[INFO] Exporting to Excel: {output_filepath}...")
        exporter = container.get_file_exporter()
        
        success = exporter.export_to_excel(str(input_filepath), str(output_filepath))
        
        if success:
            print(f"✅ SUCCESS! Your report is ready at: {output_filepath}")
            # Show size
            size = os.path.getsize(output_filepath)
            print(f"   File size: {size / 1024:.2f} KB")
        else:
            print("❌ FAILED to export to Excel. Check logs for details.")

        # 7. Cleanup
        cleanup_choice = input("\nClean up model memory? (y/n): ").lower()
        if cleanup_choice == 'y':
            model.cleanup()
            print("[SUCCESS] Model memory cleaned up.")

    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        import traceback
        traceback.print_exc()


def model_testing_menu():
    """Submenu for model testing and evaluation."""
    while True:
        print("\n" + "=" * 40)
        print("🧪 MODEL TESTING & EVALUATION")
        print("=" * 40)
        print("1. Test Local Hugging Face model")
        print("2. Test Claude model")
        print("3. Compare all available models")
        print("4. Back to main menu\n")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            print()
            test_local_model()
        elif choice == "2":
            print()
            test_claude_model()
        elif choice == "3":
            print()
            compare_models()
        elif choice == "4":
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number from 1 to 4.\n")


def learning_menu():
    """Submenu for learning resources and examples."""
    while True:
        print("\n" + "=" * 40)
        print("📖 LEARNING & EXAMPLES")
        print("=" * 40)
        print("1. Learn about LangChain concepts")
        print("2. View example code snippets")
        print("3. Back to main menu\n")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            print()
            explain_langchain_concepts()
        elif choice == "2":
            print()
            show_examples()
        elif choice == "3":
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number from 1 to 3.\n")


def context_benchmarking_menu():
    """Submenu for context-based queries and cost benchmarking."""
    while True:
        print("\n" + "=" * 40)
        print("📊 CONTEXT & BENCHMARKING")
        print("=" * 40)
        print("1. Ask with selected development context")
        print("2. Test API — no context (cost benchmark)")
        print("3. Test API — Python context (cost benchmark)")
        print("4. Back to main menu\n")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            print()
            ask_with_context()
        elif choice == "2":
            print()
            run_test_no_context()
        elif choice == "3":
            print()
            run_test_python_context()
        elif choice == "4":
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number from 1 to 4.\n")


def system_maintenance_menu():
    """Submenu for system monitoring and maintenance."""
    while True:
        print("\n" + "=" * 40)
        print("🔧 SYSTEM & MAINTENANCE")
        print("=" * 40)
        print("1. Check memory usage")
        print("2. Clean up model memory")
        print("3. View system information")
        print("4. Back to main menu\n")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            print()
            check_memory_usage()
        elif choice == "2":
            print()
            cleanup_model_memory()
        elif choice == "3":
            print()
            show_system_info()
        elif choice == "4":
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number from 1 to 4.\n")


def main():
    """
    Main function demonstrating LangChain with multiple model types.
    """
    print("\n\nLangChain Model Testing Lab")
    print("=" * 60)
    print("Goal: Test and compare different LLM models (Local, Claude)")
    print("=" * 60)

    # Show API key status
    print_api_key_status()

    # Show system info on first run
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            print("⚠️  WARNING: System memory is high (>80%).")
            print("   Consider closing other applications before testing local models.")
    except:
        pass

    # Main menu
    while True:
        print("\n" + "=" * 40)
        print("MAIN MENU")
        print("=" * 40)
        print("What would you like to do?")
        print("\n⭐ FEATURED:")
        print("1. Android Code-Gen Agent")
        print("2. LLM to Excel Agent")
        print("\n📚 TESTING & LEARNING:")
        print("3. Model Testing & Evaluation")
        print("4. Learning & Examples")
        print("\n📊 ADVANCED:")
        print("5. Context & Benchmarking")
        print("6. System & Maintenance")
        print("\n7. Exit\n")

        choice = input("Enter your choice (1-7): ").strip()

        if choice == "1":
            print()
            run_android_agent()
            print()
        elif choice == "2":
            print()
            run_hf_to_excel_agent()
            print()
        elif choice == "3":
            model_testing_menu()
            print()
        elif choice == "4":
            learning_menu()
            print()
        elif choice == "5":
            context_benchmarking_menu()
            print()
        elif choice == "6":
            system_maintenance_menu()
            print()
        elif choice == "7":
            print("\nThank you for using the LangChain Model Testing Lab!")
            print("All memory will be freed when you close this application.")
            print("Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number from 1 to 7.\n")

if __name__ == "__main__":
    # Check for required packages
    try:
        import psutil
    except ImportError:
        print("\n[INFO] psutil not installed. Some features will be limited.")
        print("[INFO] Install with: pip install psutil")
        print("[INFO] Press Enter to continue without memory monitoring...")
        input()
    
    main()