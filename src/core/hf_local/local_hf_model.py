"""
Local Hugging Face model manager for LangChain.
Designed for learning and experimentation with proper resource management.
"""

import gc
import os
import time
import torch
import psutil
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Allow non-contiguous CUDA memory allocation — prevents OOM from fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .device_utils import _select_device, _get_max_memory, _build_bnb_config
from .loading_spinner import LoadingSpinner


class LocalHuggingFaceModel:
    """
    A class to manage local Hugging Face models with LangChain.
    Designed for learning and experimentation.
    """

    def __init__(self,
                 model_id: Optional[str] = None,
                 device: str = "auto",
                 max_length: int = 512,
                 temperature: float = 0.2):
        """
        Initialize the local model.

        Args:
            model_id: Hugging Face model ID or local path
            device: "auto", "cuda", or "cpu"
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.model_id = model_id or "JetBrains/Mellum-4b-sft-kotlin"
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._pipeline = None

        print(f"\n[INFO] Initializing local model: {self.model_id}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Max length: {max_length}")
        print(f"[INFO] Temperature: {temperature}")

    def _initialize_model(self):
        """Initialize the model, tokenizer, and pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        # Limit CPU thread usage to 85% of available cores
        cpu_threads = max(1, int(os.cpu_count() * 0.85))
        torch.set_num_threads(cpu_threads)
        print(f"[INFO] CPU threads limited to {cpu_threads}/{os.cpu_count()}")

        spinner = LoadingSpinner()

        try:
            # Step 1: Load tokenizer
            print("[STEP 1] Loading tokenizer...")
            spinner.start("[LOADING] Loading tokenizer")

            # Resolve huggingface-hub cache path if necessary
            actual_model_id = self.model_id
            if os.path.isdir(self.model_id):
                # Check for any models--* subdirectory (huggingface-hub cache structure)
                for entry in os.listdir(self.model_id):
                    if entry.startswith("models--"):
                        snapshots_path = os.path.join(self.model_id, entry, "snapshots")
                        if os.path.isdir(snapshots_path):
                            snapshots = sorted(os.listdir(snapshots_path))
                            if snapshots:
                                actual_model_id = os.path.join(snapshots_path, snapshots[-1])
                                print(f"[INFO] Resolved cache path to: {actual_model_id}")
                                break

            self._tokenizer = AutoTokenizer.from_pretrained(
                actual_model_id,
                token=os.getenv("HUGGINGFACE_API_KEY")
            )

            spinner.stop("[SUCCESS] Tokenizer loaded")

            # Ensure we have a padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Step 2: Load model
            print("\n[STEP 2] Loading model...")
            print("[INFO] This may take a moment, especially for large models...")

            # Select device and dtype
            device, dtype = _select_device()

            # Note: The transformers library will show its own progress bar
            # for loading checkpoint shards, so we don't need our spinner here

            # Load model with device-specific configuration
            if device == "cuda":
                max_mem = _get_max_memory(fraction=0.75)
                print(f"[INFO] Memory limits: {max_mem}")
                print("[INFO] Large models will offload to CPU if needed (slower but works)")

                # Try loading with 4-bit quantization first
                try:
                    print("[INFO] Applying 4-bit quantization to reduce VRAM usage...")
                    bnb_config = _build_bnb_config()

                    self._model = AutoModelForCausalLM.from_pretrained(
                        actual_model_id,
                        torch_dtype=dtype,
                        device_map="auto",
                        quantization_config=bnb_config,
                        max_memory=max_mem,
                        token=os.getenv("HUGGINGFACE_API_KEY")
                    )
                except (ValueError, RuntimeError) as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "dispatched on the cpu or the disk" in error_msg:
                        print(f"[WARNING] 4-bit quantization failed: {e}")
                        print("[WARNING] Doing full GPU cleanup before fallback...")

                        # Full cleanup before retry
                        if self._model is not None:
                            del self._model
                            self._model = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                        # Recalculate max_memory AFTER cleanup
                        max_mem = _get_max_memory(fraction=0.75)
                        print(f"[WARNING] Falling back to unquantized float16. Memory after cleanup: {max_mem}")

                        self._model = AutoModelForCausalLM.from_pretrained(
                            actual_model_id,
                            torch_dtype=dtype,
                            device_map="auto",
                            max_memory=max_mem,
                            token=os.getenv("HUGGINGFACE_API_KEY")
                        )
                    else:
                        raise
            elif device == "mps":
                # For MPS: don't use device_map, load normally then move to device
                self._model = AutoModelForCausalLM.from_pretrained(
                    actual_model_id,
                    torch_dtype=dtype,
                    token=os.getenv("HUGGINGFACE_API_KEY")
                )
                self._model = self._model.to(device)
            else:
                # For CPU: load normally without device_map
                self._model = AutoModelForCausalLM.from_pretrained(
                    actual_model_id,
                    torch_dtype=dtype,
                    token=os.getenv("HUGGINGFACE_API_KEY")
                )

            # Wait a moment to let the transformers progress bar finish
            time.sleep(0.5)
            print("[SUCCESS] Model loaded")

            # Step 3: Create pipeline
            print("\n[STEP 3] Creating text-generation pipeline...")
            spinner.start("[INITIALIZING] Setting up pipeline")

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )

            spinner.stop("[SUCCESS] Pipeline ready")

            print("\n" + "=" * 50)
            print("[SUCCESS] Model initialized and ready for use!")
            print("=" * 50)
            return self._pipeline

        except Exception as e:
            if 'spinner' in locals():
                spinner.stop("[ERROR] Initialization failed!")
            print(f"[ERROR] Failed to initialize model: {e}")
            raise

    def generate(self, prompt: str, skip_prompt: bool = False, **kwargs) -> Optional[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            skip_prompt: Skip user confirmation
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        spinner = LoadingSpinner()

        try:
            if self._pipeline is None:
                self._initialize_model()

            # Initialize TokenManager if not already done
            if not hasattr(self, 'token_manager'):
                from ..utils import TokenManager
                self.token_manager = TokenManager()

            # Format the prompt for better results
            formatted_prompt = self._format_prompt(prompt)

            # Token analysis before generation
            prompt_tokens = len(self._tokenizer.encode(formatted_prompt))
            model_name = f"huggingface/{self.model_id.split('/')[-1]}"  # Use as identifier for logging

            print("\n[TokenManager] Prompt Analysis:")
            print(f"- Input tokens: {prompt_tokens}")
            print(f"- Model: {model_name}")

            # Check if user wants to continue
            if not skip_prompt and not prompt_continue():
                print("[INFO] Generation skipped by user.")
                return None

            print(f"\n[PROMPT] {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            # Start spinner
            spinner.start("[GENERATING] Processing your request")

            try:
                # Generate response
                result = self._pipeline(
                    formatted_prompt,
                    max_new_tokens=kwargs.get('max_new_tokens', self.max_length),
                    temperature=kwargs.get('temperature', self.temperature),
                    do_sample=True,
                    num_return_sequences=1
                )
            finally:
                # Always stop spinner
                spinner.stop("[SUCCESS] Response generated!")

            # Extract the generated text
            generated_text = result[0]['generated_text']

            # Remove the input prompt from the response
            if formatted_prompt in generated_text:
                response = generated_text[len(formatted_prompt):].strip()
            else:
                response = generated_text

            # Token analysis after generation
            response_tokens = len(self._tokenizer.encode(response))
            print("\n[TokenManager] Response Analysis:")
            print(f"- Output tokens: {response_tokens}")

            # Log usage (using placeholder costs since local inference is free)
            self.token_manager.log_usage(model_name, prompt_tokens, "hf_prompt", is_output=False)
            self.token_manager.log_usage(model_name, response_tokens, "hf_response", is_output=True)

            # Show summary
            summary = self.token_manager.get_usage_summary()
            print("\n[TokenManager] Session Summary:")
            print(f"- Total tokens used: {summary['total_tokens']}")
            print(f"- Total estimated cost: ${summary['total_cost']:.6f}")

            return response

        except Exception as e:
            if 'spinner' in locals():
                spinner.stop("[ERROR] Generation failed!")
            print(f"[ERROR] Generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for better model performance."""
        # Simple formatting - you can customize this based on your model
        if "Kotlin" in prompt or "code" in prompt.lower():
            return f"Write Kotlin code for: {prompt}\n\nHere's the Kotlin code:"
        else:
            return f"Question: {prompt}\n\nAnswer:"

    def cleanup(self):
        """
        Explicitly clean up model resources to free memory.
        This is useful when you're done with the model but want to keep running other code.
        """
        print("\n[INFO] Cleaning up model resources...")

        # Clear pipeline
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        # Clear model
        if self._model is not None:
            del self._model
            self._model = None

        # Clear tokenizer
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Force garbage collection
        gc.collect()

        # Clear device-specific caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print("[SUCCESS] Model resources cleaned up!")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically clean up."""
        self.cleanup()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}

        return {
            "model_id": self.model_id,
            "device": str(self._model.device),
            "dtype": str(self._model.dtype),
            "parameters": sum(p.numel() for p in self._model.parameters()),
            "status": "loaded"
        }

    def test_connection(self) -> bool:
        """Test if the model can generate a simple response."""
        try:
            test_prompt = "Say 'Hello, World!' in Kotlin."
            print(f"\n[TEST] Testing model with prompt: '{test_prompt}'")

            response = self.generate(test_prompt, max_new_tokens=50, skip_prompt=True)

            print(f"\n[TEST] Response: {response}")
            return True

        except Exception as e:
            print(f"[TEST FAILED] {e}")
            return False
