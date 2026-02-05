"""
Module for working with local Hugging Face models in LangChain.
Focuses on educational aspects and proper error handling.
"""

import os
import torch
import threading
import time
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from typing import Optional, Dict, Any

class LoadingSpinner:
    """Display a loading spinner animation."""
    
    def __init__(self, message: str = "Loading", delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.busy = False
        self.spinner_thread = None
    
    def spin(self):
        """Spin the loading animation."""
        while self.busy:
            for char in self.spinner_chars:
                if not self.busy:
                    break
                sys.stdout.write(f'\r{self.message} {char}')
                sys.stdout.flush()
                time.sleep(self.delay)
    
    def start(self, message: Optional[str] = None):
        """Start the spinner animation."""
        if message:
            self.message = message
        self.busy = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def stop(self, final_message: str = ""):
        """Stop the spinner animation."""
        self.busy = False
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)
        # Clear the line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        if final_message:
            print(f"\r{final_message}")
        sys.stdout.flush()

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
        
        spinner = LoadingSpinner()
        
        try:
            # Step 1: Load tokenizer
            print("[STEP 1] Loading tokenizer...")
            spinner.start("[LOADING] Loading tokenizer")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=os.getenv("HUGGINGFACE_API_KEY")
            )
            
            spinner.stop("[SUCCESS] Tokenizer loaded")
            
            # Ensure we have a padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Step 2: Load model
            print("\n[STEP 2] Loading model...")
            print("[INFO] This may take a moment, especially for large models...")
            
            # Note: The transformers library will show its own progress bar
            # for loading checkpoint shards, so we don't need our spinner here
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.device,
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
                from .token_utils import TokenManager
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
        import gc
        gc.collect()
        
        # Clear PyTorch CUDA cache if using GPU
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
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
        spinner = LoadingSpinner()
        
        try:
            test_prompt = "Say 'Hello, World!' in Kotlin."
            print(f"\n[TEST] Testing model with prompt: '{test_prompt}'")
            
            spinner.start("[TESTING] Running test generation")
            
            response = self.generate(test_prompt, max_new_tokens=50)
            
            spinner.stop("[SUCCESS] Test completed")
            
            print(f"\n[TEST] Response: {response}")
            return True
            
        except Exception as e:
            if 'spinner' in locals():
                spinner.stop("[ERROR] Test failed!")
            print(f"[TEST FAILED] {e}")
            return False
    
    def generate_with_progress(self, prompt: str, **kwargs) -> str:
        """
        Alternative generate method that shows estimated progress.
        This is useful for longer generations.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        print(f"\n[PROMPT] {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print("[INFO] Starting generation...")
        
        # Show some initial feedback
        print("[GENERATING] ", end="", flush=True)
        
        # Simple dots animation
        def show_dots():
            for i in range(10):
                if i > 0:
                    print(".", end="", flush=True)
                time.sleep(0.3)
            print("")
        
        # Start dots animation in background
        dots_thread = threading.Thread(target=show_dots)
        dots_thread.daemon = True
        dots_thread.start()
        
        try:
            if self._pipeline is None:
                self._initialize_model()
            
            # Format the prompt for better results
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate response
            result = self._pipeline(
                formatted_prompt,
                max_new_tokens=kwargs.get('max_new_tokens', self.max_length),
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=True,
                num_return_sequences=1
            )
            
            # Wait for dots animation to finish if it hasn't
            dots_thread.join(timeout=1.0)
            
            print("[SUCCESS] Generation complete!")
            
            # Extract the generated text
            generated_text = result[0]['generated_text']
            
            # Remove the input prompt from the response
            if formatted_prompt in generated_text:
                response = generated_text[len(formatted_prompt):].strip()
            else:
                response = generated_text
            
            return response
            
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            return f"Error generating response: {str(e)}"