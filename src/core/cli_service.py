"""
Interactive CLI service for the LangChain application.
Extracts all UI and orchestration logic from main.py following SRP.
"""

from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path
import gc
import psutil
from dotenv import load_dotenv

if TYPE_CHECKING:
    from .dependency_injection import DIContainer

from .config import *
from .interfaces import ILanguageModel, ModelConfig, GenerationStrategy
from .models.model_factory import ModelFactory
from .services import ConfigurationManager, ConsoleUserInteraction
from .model_comparison import ModelComparison
from .prompt_manager import PromptManager
from .token_utils import TokenManager
from .exporters import ExcelExporter
from .exceptions import *
import torch

class InteractiveCLI:
    def __init__(self, container: 'DIContainer'):
        self.container = container
        self.factory = container.get_model_factory()
        self.user_interaction = container.get_user_interaction()
        self.config_manager = container.get_config_manager()
        self.logger = container.get_logging_service().get_logger("cli")
        self.prompt_manager = container.get_prompt_manager()
        self.token_manager = container.get_token_manager()
        self.model_comparison = ModelComparison()

    def print_api_key_status(self):
        """Display the status of API keys."""
        keys = {
            "Hugging Face": self.config_manager.get_api_key("huggingface"),
            "Anthropic": self.config_manager.get_api_key("anthropic")
        }
    
        print("\n" + "=" * 40)
        print("API Key Status:")
        print("-" * 40)
        for name, key in keys.items():
            is_ok = key and "your-" not in key.lower() and key.strip() != ""
            print(f"{name:13}: {'[OK] Configured' if is_ok else '[X] Missing'}")
        print("=" * 40 + "\n")

    def check_memory_usage(self):
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

    def cleanup_model_memory(self):
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
                print("[INFO] No GPU detected")
        
            # Final memory check
            if initial_memory:
                try:
                    process = psutil.Process(os.getpid())
                    final_memory = process.memory_info().rss
                    savings = (initial_memory - final_memory) / 1024 / 1024
                    print(f"\n[INFO] Memory savings: {savings:.1f} MB")
                except:
                    pass
        
            print("\n[SUCCESS] Memory cleanup completed!")
        
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")

    def test_claude_model(self):
        """Test the Claude model with different prompts."""
        print("[TEST] Claude Model Demo")
        print("-" * 40)
        config = ModelConfig(
            model_name=DEFAULT_MODEL,
            system_message="You are a helpful assistant.",
            max_tokens=512,
            temperature=0.2,
        )
        model = self.factory.create_model("anthropic", config)
        self._run_test_loop(model, TEST_PROMPTS)
        print("\n" + "=" * 40)
        print("Claude Model Testing Complete!")
        print("=" * 40)

    def test_local_model(self):
        """Test local HF model."""
        print("\n🧪 Testing Local HF Model...")
        model_name = "microsoft/DialoGPT-medium"  # or prompt for
        config = ModelConfig(
            model_name=model_name,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.7,
        )
        model = self.factory.create_model("huggingface", config)
        test_prompts = TEST_PROMPTS  # or local specific
        self._run_test_loop(model, test_prompts)

    def _run_test_loop(self, model: ILanguageModel, prompts):
        """Common test loop."""
        for i, prompt_info in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Testing: {prompt_info['name']}")
            print(f"Prompt: {prompt_info['prompt']}")
            print("-" * 40)
            result = model.generate(prompt_info["prompt"])
            print(f"Response:\n{result.content}")
            if i < len(prompts):
                if not self.user_interaction.prompt_continue():
                    break

    def run_android_agent(self):
        """Android agent."""
        print("\n🚀 Android Code-Gen Agent")
        print("=" * 60)
        # TODO: Move full logic here using DI
        input("Press Enter to return to menu...")

    # Add other methods...

    def run_menu(self):
        """Main interactive menu."""
        self.print_api_key_status()
        while True:
            print("\n" + "=" * 60)
            print("MAIN MENU")
            print("=" * 60)
            print("What would you like to do?")
            print("")
            print("⭐ FEATURED:")
            print("1. Android Code-Gen Agent")
            print("2. LLM to Excel Agent")
            print("3. JSON to Excel (no LLM - paste your JSON in data/default_json_for_excel_convertion.json)")
            print("")
            print("📚 TESTING & LEARNING:")
            print("4. Model Testing & Evaluation")
            print("5. Learning & Examples")
            print("")
            print("📊 ADVANCED:")
            print("6. Context & Benchmarking")
            print("7. System & Maintenance")
            print("")
            print("8. Exit")
            print("")
            choice = input("Enter your choice (1-8): ").strip()
            if choice == "1":
                self.run_android_agent()
            elif choice == "2":
                self.llm_to_excel_agent()
            elif choice == "3":
                self.json_to_excel()
            elif choice == "4":
                self.model_testing_menu()
            elif choice == "5":
                self.learning_examples_menu()
            elif choice == "6":
                self.context_benchmarking_menu()
            elif choice == "7":
                self.system_maintenance_menu()
            elif choice == "8":
                print("\n👋 Thanks for using LangChain Model Lab!")
                break
            else:
                print("\n[ERROR] Invalid choice. Please select 1-8.")
                input("Press Enter to continue...")
        self.cleanup_model_memory()
