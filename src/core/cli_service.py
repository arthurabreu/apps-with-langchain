"""
Interactive CLI service for the LangChain application.

Android Analogy:
- This is your 'MainActivity' logic or a 'Presenter/ViewModel' that handles 
  user input and coordinates different services.
- It orchestrates the flow: showing menus, calling models, and displaying results.
"""

from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path
import os
import sys
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
        """
        Initialize the CLI with all needed services from the DI container.
        
        Android Analogy: Similar to how you'd get services from a 
        Dagger Component or Koin 'get()'.
        """
        self.container = container
        self.factory = container.get_model_factory()
        self.user_interaction = container.get_user_interaction()
        self.config_manager = container.get_config_manager()
        self.logger = container.get_logging_service().get_logger("cli")
        self.prompt_manager = container.get_prompt_manager()
        self.token_manager = container.get_token_manager()
        self.model_comparison = ModelComparison()

    def print_api_key_status(self):
        """
        Check and print if API keys are correctly set in .env.
        Like a 'checkPermissions' or 'checkPrerequisites' step.
        """
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

    def prompt_for_task(self, agent_name: str = "Agent") -> str:
        """
        Prompt user to choose between loading a task from file or typing manually.
        
        Android Analogy: Similar to selecting an Intent source or picking a file 
        via an ActivityResultLauncher.
        
        Args:
            agent_name: Name of the agent for display (e.g., "Android Agent").

        Returns:
            The task/prompt string.
        """
        files_dir = Path("files")

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
                            return self.prompt_for_task(agent_name)
                        else:
                            print("Invalid selection. Try again.\n")
                    else:
                        print("Invalid input. Try again.\n")
            elif choice == "2":
                print("\n📝 Task Description:")
                task = input("Enter what you would like to generate: ").strip()
                if not task:
                    print("[ERROR] Task cannot be empty.")
                    return self.prompt_for_task(agent_name)
                return task
            else:
                print("[ERROR] Invalid choice. Please select 1 or 2.")
                return self.prompt_for_task(agent_name)
        else:
            # No script files available, just ask for input
            print("\n📝 Task Description:")
            task = input("Enter what you would like to generate: ").strip()
            if not task:
                print("[ERROR] Task cannot be empty.")
                return self.prompt_for_task(agent_name)
            return task

    def run_android_agent(self):
        """
        Launch the Android code-gen agent.
        Orchestrates task selection and hands off to the specialized agent logic.
        """
        print("\n🚀 Android Code-Gen Agent")
        print("=" * 60)
        try:
            # Add src to path so we can import from other modules
            sys.path.append(str(Path(__file__).parent.parent))
            
            # Note: We keep the import here as it might have heavy dependencies 
            # or be part of a separate module.
            from android_agent.cli import main as android_main

            # Get task from user (file or manual input)
            task = self.prompt_for_task("Android Code-Gen Agent")
            if not task:
                return

            # Pass task to the android agent
            android_main(task=task)
        except Exception as e:
            print(f"[ERROR] Failed to launch Android agent: {e}")
        
        input("\nPress Enter to return to menu...")

    def llm_to_excel_agent(self):
        """
        Launch the LLM to Excel Agent.
        Uses local models to generate data and exporters to create Excel files.
        """
        print("\n🤖 LLM TO EXCEL AGENT (Local Hugging Face)")
        print("=" * 60)

        try:
            # 1. Select HF model (logic usually in hf_model_manager)
            from core.hf_model_manager import select_hf_model
            from core.services import get_brazil_time
            
            result = select_hf_model()
            if result is None:
                return

            model_id, model_folder = result
            print(f"✓ Using model: {model_id}\n")

            # 2. Get task/prompt
            task = self.prompt_for_task("LLM to Excel Agent")
            if not task:
                return

            print("💡 Tip: Return valid JSON for the best Excel formatting.\n")

            # 3. Choose output format
            print("\n📄 Select Response Format:")
            print("1. JSON (Recommended) [DEFAULT]")
            print("2. Plain Text / Markdown")
            print("3. Convert existing JSON file (no LLM)")
            format_choice = input("Select [1, 2 or 3, default=1]: ").strip()

            if format_choice == "3":
                self.json_to_excel()
                return

            is_json = format_choice != "2"
            file_ext = ".json" if is_json else ".md"
            if is_json:
                task += "\n\nIMPORTANT: Return ONLY a valid JSON object or list. No other text."

            # 4. Initialize and run model
            config = ModelConfig(
                model_name=str(model_folder),
                temperature=0.7,
                max_tokens=1024
            )

            print("\n[INFO] Initializing model...")
            model = self.factory.create_model("huggingface", config)

            print("\n[INFO] Generating response...")
            gen_result = model.generate(task, skip_prompt=True)
            content = gen_result.content.strip()

            # 5. Save response
            responses_dir = Path(RESPONSES_DIR) / "excel_exports"
            responses_dir.mkdir(parents=True, exist_ok=True)

            timestamp = get_brazil_time().strftime("%Y%m%d_%H%M%S")
            input_filepath = responses_dir / f"llm_output_{timestamp}{file_ext}"

            with open(input_filepath, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✓ Response saved to: {input_filepath}")

            # 6. Export to Excel
            output_filepath = responses_dir / f"llm_report_{timestamp}.xlsx"
            print(f"\n[INFO] Exporting to Excel: {output_filepath}...")
            
            exporter = self.container.get_file_exporter()
            success = exporter.export_to_excel(str(input_filepath), str(output_filepath))

            if success:
                print(f"✅ SUCCESS! Report: {output_filepath}")
            else:
                print("❌ FAILED to export to Excel.")

            # 7. Cleanup
            if input("\nClean up model memory? (y/n): ").lower() == 'y':
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                self.cleanup_model_memory()

        except Exception as e:
            print(f"[ERROR] Agent failed: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to return to menu...")

    def json_to_excel(self):
        """Convert a local JSON file directly to Excel without LLM usage."""
        print("\n📊 JSON TO EXCEL CONVERTER")
        print("=" * 60)
        
        default_json = Path("data/default_json_for_excel_convertion.json")

        if not default_json.exists():
            print(f"\n[X] File not found: {default_json}")
            print("Please create it and paste your JSON data.")
            return

        print(f"✓ Found: {default_json}")
        
        from core.services import get_brazil_time
        responses_dir = Path(RESPONSES_DIR) / "excel_exports"
        responses_dir.mkdir(parents=True, exist_ok=True)

        timestamp = get_brazil_time().strftime("%Y%m%d_%H%M%S")
        output_filepath = responses_dir / f"manual_report_{timestamp}.xlsx"

        exporter = self.container.get_file_exporter()
        success = exporter.export_to_excel(str(default_json), str(output_filepath))

        if success:
            print(f"✅ SUCCESS! Report: {output_filepath}")
        else:
            print("❌ FAILED to export.")
            
        input("\nPress Enter to return to menu...")

    def model_testing_menu(self):
        """Submenu for model testing and evaluation."""
        while True:
            print("\n" + "=" * 40)
            print("🧪 MODEL TESTING & EVALUATION")
            print("=" * 40)
            print("1. Test Claude Model (Interactive)")
            print("2. Test Local HF Model")
            print("3. Compare All Available Models")
            print("4. Back to main menu\n")

            choice = input("Enter choice (1-4): ").strip()
            if choice == "1":
                self.test_claude_model()
            elif choice == "2":
                self.test_local_model()
            elif choice == "3":
                self.compare_models()
            elif choice == "4":
                break
            else:
                print("\n[ERROR] Invalid choice.")

    def compare_models(self):
        """
        Compare performance and output quality across multiple providers.
        """
        print("\n⚖️ MODEL COMPARISON TOOL")
        # Logic adapted from original main.py
        try:
            from core.model_comparison import ModelComparison
            
            # Simple check for available providers
            providers = []
            if self.config_manager.get_api_key("anthropic"): providers.append("Claude")
            # We assume HF is available if package is there
            providers.append("Local HF")
            
            print(f"[INFO] Models for comparison: {', '.join(providers)}")
            
            # Use defaults or custom
            prompts = TEST_PROMPTS
            if input("\nUse default test prompts? (y/n): ").lower() != 'y':
                prompts = []
                print("Enter 'done' to finish.")
                while True:
                    name = input("Prompt name: ").strip()
                    if name.lower() == 'done': break
                    txt = input("Prompt text: ").strip()
                    prompts.append({"name": name, "prompt": txt})
                if not prompts: prompts = TEST_PROMPTS

            comparison = ModelComparison()
            comparison.run_comparison(prompts)
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
        
        input("\nPress Enter to return...")

    def learning_examples_menu(self):
        """Submenu for learning resources and code examples."""
        while True:
            print("\n" + "=" * 40)
            print("📚 LEARNING & EXAMPLES")
            print("=" * 40)
            print("1. Explain LangChain Concepts")
            print("2. Show Example Code Snippets")
            print("3. Back to main menu\n")

            choice = input("Enter choice (1-3): ").strip()
            if choice == "1":
                self.explain_langchain_concepts()
            elif choice == "2":
                self.show_examples()
            elif choice == "3":
                break
            else:
                print("\n[ERROR] Invalid choice.")

    def explain_langchain_concepts(self):
        """Explains Core LangChain concepts for Android/Python developers."""
        concepts = {
            "Chains": "Sequences of operations. In Android, think of this as a Flow or Rx Pipeline.",
            "Prompts": "Templates for LLM instructions. Similar to String resources with placeholders.",
            "Models": "The 'Engine' (Claude, GPT, Llama). Think of them as specialized Repositories.",
            "Memory": "Persistence for conversation history. Similar to SharedPreferences or Room for UI state."
        }
        for concept, desc in concepts.items():
            print(f"\n💡 {concept}:")
            print(f"   {desc}")
            input("\nPress Enter for next...")

    def show_examples(self):
        """Displays practical examples of library usage."""
        print("\n📝 EXAMPLE: Simple Chain")
        print("```python\nchain = prompt | model | output_parser\nresult = chain.invoke({'input': 'Hi'})\n```")
        input("\nPress Enter to return...")

    def context_benchmarking_menu(self):
        """Submenu for context-based tests and cost benchmarks."""
        while True:
            print("\n" + "=" * 40)
            print("📊 CONTEXT & BENCHMARKING")
            print("=" * 40)
            print("1. Select Context & Ask Question")
            print("2. Run Test: Direct API (No Context)")
            print("3. Run Test: With Python Expert Context")
            print("4. Back to main menu\n")

            choice = input("Enter choice (1-4): ").strip()
            if choice == "1":
                self.ask_with_context()
            elif choice == "2":
                self.run_test_no_context()
            elif choice == "3":
                self.run_test_python_context()
            elif choice == "4":
                break

    def select_context(self):
        """Helper to pick a system context from config."""
        print("\n📁 Select Context:")
        contexts = list(CONTEXT_DEFAULTS.keys())
        for i, ctx in enumerate(contexts, 1):
            print(f"{i}. {ctx}")
        
        try:
            idx = int(input("Select [number]: ")) - 1
            if 0 <= idx < len(contexts):
                key = contexts[idx]
                data = CONTEXT_DEFAULTS[key]
                # In real app, we might have system messages mapped elsewhere
                return key, "You are an expert.", data['max_tokens'], data['temperature']
        except:
            pass
        return None

    def ask_with_context(self):
        """Ask Claude a question using a predefined system context."""
        ctx_info = self.select_context()
        if not ctx_info: return
        
        key, sys_msg, tokens, temp = ctx_info
        question = input(f"\n[{key}] Enter question: ").strip()
        if not question: return

        config = ModelConfig(
            model_name=DEFAULT_MODEL,
            system_message=sys_msg,
            max_tokens=tokens,
            temperature=temp
        )
        model = self.factory.create_model("anthropic", config)
        print("\n[INFO] Thinking...")
        result = model.generate(question)
        print(f"\nResponse:\n{result.content}")
        input("\nPress Enter...")

    def run_test_no_context(self):
        """Benchmark test with no specific system instructions."""
        self.test_claude_model()

    def run_test_python_context(self):
        """Benchmark test with specialized Python instructions."""
        print("\n🧪 Running with Python Context...")
        # Simplified for brevity, similar to test_claude_model but with specific context
        self.test_claude_model()

    def system_maintenance_menu(self):
        """Submenu for monitoring and cleanup."""
        while True:
            print("\n" + "=" * 40)
            print("🔧 SYSTEM & MAINTENANCE")
            print("=" * 40)
            print("1. Check Memory Usage")
            print("2. Clean up Model Memory")
            print("3. Show System Info")
            print("4. Back to main menu\n")

            choice = input("Enter choice (1-4): ").strip()
            if choice == "1":
                self.check_memory_usage()
            elif choice == "2":
                self.cleanup_model_memory()
            elif choice == "3":
                self.show_system_info()
            elif choice == "4":
                break

    def show_system_info(self):
        """Displays environment and hardware details."""
        print("\n🖥️ SYSTEM INFO")
        print(f"OS: {os.name}")
        print(f"CPUs: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        input("\nPress Enter...")

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
