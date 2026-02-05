"""
Main application file for LangChain project.
Focused on testing local Hugging Face models with LangChain.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def print_api_key_status():
    """Display the status of API keys."""
    keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
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

def test_local_model():
    """Test the local Hugging Face model with different prompts."""
    print("[TEST] Local Hugging Face Model Demo")
    print("-" * 40)
    
    try:
        from core.langchain_huggingface_local import LocalHuggingFaceModel
        
        # Check memory before loading model
        print("\n[INFO] Checking memory before loading model...")
        check_memory_usage()
        
        # Initialize local model
        print("\n[INFO] Initializing local model (this may take a moment)...")
        local_model = LocalHuggingFaceModel()
        
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
                response = local_model.generate(test['prompt'])
                print(f"Response:\n{response}")            
            except Exception as e:
                print(f"Error: {e}")
            
            if i < len(test_prompts):
                if not prompt_continue_or_skip():
                    print("\n[INFO] Skipping remaining tests...")
                    break
        
        print("\n" + "=" * 40)
        print("Local Model Testing Complete!")
        
        # Ask if user wants to clean up
        cleanup_choice = input("\nClean up model memory? (y/n): ").lower()
        if cleanup_choice == 'y':
            local_model.cleanup()
            del local_model
            import gc
            gc.collect()
            print("[SUCCESS] Model memory cleaned up!")
            check_memory_usage()
        else:
            print("[INFO] Model remains loaded in memory.")
            print("[INFO] You can clean up later from the main menu.")
        
        print("=" * 40)
        
    except ImportError as e:
        print(f"[ERROR] Missing dependencies for local model: {e}")
        print("Install with: pip install transformers torch langchain-huggingface")
    except Exception as e:
        print(f"[ERROR] Failed to initialize local model: {e}")

def test_openai_model():
    """Test the OpenAI model with different prompts."""
    print("[TEST] OpenAI Model Demo")
    print("-" * 40)
    
    # Check if OpenAI API key is configured
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or "your-" in openai_key.lower():
        print("\n[ERROR] OpenAI API key not configured.")
        print("Add your OPENAI_API_KEY to the .env file")
        print("Format: OPENAI_API_KEY=your-actual-key-here")
        return
    
    try:
        from core.openai_model import OpenAIModel
        
        # Initialize OpenAI model
        openai_model = OpenAIModel()
        
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
                response = openai_model.generate(test['prompt'])
                print(f"Response:\n{response}")
            except Exception as e:
                print(f"Error: {e}")
            
            if i < len(test_prompts):
                input("\nPress Enter to continue to next test...")
        
        print("\n" + "=" * 40)
        print("OpenAI Model Testing Complete!")
        print("=" * 40)
        
    except ImportError as e:
        print(f"[ERROR] Missing dependencies for OpenAI: {e}")
        print("Install with: pip install langchain-openai")
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI model: {e}")

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
        from core.claude_model import ClaudeModel
        
        # Initialize Claude model
        claude_model = ClaudeModel()
        
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
                response = claude_model.generate(test['prompt'])
                print(f"Response:\n{response}")
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
            from datetime import datetime
            import json
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{timestamp}.json"
            
            # Prepare results
            results = {
                "timestamp": datetime.now().isoformat(),
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
            print("   Consider using cloud models (OpenAI/Claude) instead.")
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

def main():
    """
    Main function demonstrating LangChain with multiple model types.
    """
    print("LangChain Model Testing Lab")
    print("=" * 60)
    print("Goal: Test and compare different LLM models (Local, OpenAI, Claude)")
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
        print("1. Test Local Hugging Face model")
        print("2. Test OpenAI model")
        print("3. Test Claude model")
        print("4. Compare all available models")
        print("5. Learn about LangChain concepts")
        print("6. View example code snippets")
        print("7. Check memory usage")
        print("8. Clean up model memory")
        print("9. View system information")
        print("10. Exit")
        
        choice = input("\nEnter your choice (1-10): ").strip()
        
        if choice == "1":
            test_local_model()
        elif choice == "2":
            test_openai_model()
        elif choice == "3":
            test_claude_model()
        elif choice == "4":
            compare_models()
        elif choice == "5":
            explain_langchain_concepts()
        elif choice == "6":
            show_examples()
        elif choice == "7":
            check_memory_usage()
        elif choice == "8":
            cleanup_model_memory()
        elif choice == "9":
            show_system_info()
        elif choice == "10":
            print("\nThank you for using the LangChain Model Testing Lab!")
            print("All memory will be freed when you close this application.")
            print("Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter a number from 1 to 10.")
        
        # Ask if user wants to continue
        if choice != "10":
            continue_choice = input("\nReturn to main menu? (y/n): ").lower()
            if continue_choice != 'y':
                print("\nExiting. Goodbye!")
                print("[INFO] Remember: Close VS Code to free all memory.")
                break

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