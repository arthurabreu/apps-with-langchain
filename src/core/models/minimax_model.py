"""
MiniMax-M2.1 model implementation following SOLID principles.
Uses local Hugging Face transformers library for inference.
"""

import logging
import torch
import os
import psutil
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..interfaces import ILanguageModel, ITokenManager, IUserInteraction, ModelConfig, GenerationResult
from ..exceptions import ModelConfigurationError, GenerationError


def _select_device() -> Tuple[str, torch.dtype]:
    """
    Detect and select the best available device for model inference.

    Returns:
        Tuple of (device_name, dtype)
        - device_name: "cuda", "mps", or "cpu"
        - dtype: torch.float16 for accelerated devices, torch.float32 for CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_name = f"cuda (GPU: {torch.cuda.get_device_name(0)})"
        print(f"[INFO] CUDA detected: Using GPU acceleration (device: {device}, dtype: {dtype})")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        device_name = "mps (Apple Silicon)"
        print(f"[INFO] MPS detected: Using Apple Silicon acceleration (device: {device}, dtype: {dtype})")
    else:
        device = "cpu"
        dtype = torch.float32
        device_name = "CPU"
        print(f"[INFO] No GPU detected: Using CPU (device: {device}, dtype: {dtype})")

    return device, dtype


def _get_max_memory(fraction: float = 0.85) -> dict:
    """
    Calculate max_memory dict to cap GPU and CPU memory at fraction of available.

    Args:
        fraction: Fraction of total available memory to use (e.g., 0.85 = 85%)

    Returns:
        Dictionary suitable for HuggingFace transformers device_map="auto"
    """
    mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            mem[i] = f"{int(total * fraction / (1024**2))}MiB"
    ram = psutil.virtual_memory().total
    mem["cpu"] = f"{int(ram * fraction / (1024**2))}MiB"
    return mem


def _build_bnb_config():
    """Build BitsAndBytesConfig for 4-bit quantization."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ValueError("transformers>=4.30 required for BitsAndBytesConfig")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class MiniMaxModel(ILanguageModel):
    """MiniMax-M2.1 model implementation with dependency injection."""

    DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-M2.1"

    def __init__(
        self,
        config: ModelConfig,
        token_manager: ITokenManager,
        user_interaction: IUserInteraction,
        logger: logging.Logger,
        cost_tracker: Optional[Any] = None
    ):
        """
        Initialize MiniMax model with dependencies.

        Args:
            config: Model configuration
            token_manager: Token management service
            user_interaction: User interaction service
            logger: Logger instance
            cost_tracker: Cost tracking service (optional)
        """
        self.token_manager = token_manager
        self.user_interaction = user_interaction
        self.logger = logger
        self.cost_tracker = cost_tracker
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None

        super().__init__(config)
        self._initialize_model()

    def _validate_config(self) -> None:
        """Validate the model configuration."""
        if not self.config.model_name:
            # Use default if not specified
            self.config.model_name = self.DEFAULT_MODEL_ID

        if not (0.0 <= self.config.temperature <= 2.0):
            raise ModelConfigurationError(
                f"Temperature must be between 0.0 and 2.0, got {self.config.temperature}"
            )

        if self.config.max_tokens < 1:
            raise ModelConfigurationError(
                f"Max tokens must be positive, got {self.config.max_tokens}"
            )

        self.logger.info(f"Configuration validated for model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the model, tokenizer, and device."""
        try:
            self.logger.info(f"Initializing MiniMax model: {self.config.model_name}")

            # Limit CPU thread usage to 85% of available cores
            cpu_threads = max(1, int(os.cpu_count() * 0.85))
            torch.set_num_threads(cpu_threads)
            self.logger.info(f"CPU threads limited to {cpu_threads}/{os.cpu_count()}")

            # Select device and dtype
            self._device, self._dtype = _select_device()
            
            # Get HF token if available
            hf_token = os.getenv("HUGGINGFACE_API_KEY")
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.user_interaction.display_info("[STEP 1/3] Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                token=hf_token
            )
            
            # Ensure we have a padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self.logger.info("Tokenizer loaded successfully")
            self.user_interaction.display_info("[STEP 2/3] Loading model (this may take a few moments)...")
            
            # Load model with device-specific configuration
            if self._device == "cuda":
                # Use 90% of VRAM to allow larger models with CPU offloading
                max_mem = _get_max_memory(fraction=0.90)
                self.logger.info(f"Memory limits: {max_mem}")
                self.logger.info("Large models will offload to CPU if needed (slower but works)")

                # Try loading with 4-bit quantization first
                try:
                    self.logger.info("Applying 4-bit quantization to reduce VRAM usage...")
                    bnb_config = _build_bnb_config()

                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self._dtype,
                        device_map="auto",
                        quantization_config=bnb_config,
                        max_memory=max_mem,
                        trust_remote_code=True,
                        token=hf_token
                    )
                except ValueError as e:
                    # 4-bit quantization failed (likely model too large for GPU offloading)
                    # Fall back to unquantized with CPU offloading
                    if "dispatched on the CPU or the disk" in str(e):
                        self.logger.warning("4-bit quantization incompatible with CPU offloading. Falling back to unquantized model with float16.")
                        self.logger.warning("This may use more VRAM but will work with CPU offloading.")

                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.config.model_name,
                            torch_dtype=self._dtype,  # float16 or bfloat16
                            device_map="auto",
                            max_memory=max_mem,
                            trust_remote_code=True,
                            token=hf_token
                        )
                    else:
                        raise
            elif self._device == "mps":
                # For MPS: don't use device_map, load normally then move to device
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    token=hf_token
                )
                self._model = self._model.to(self._device)
            else:
                # For CPU: load normally without device_map
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    token=hf_token
                )
            
            self.logger.info("Model loaded successfully")
            self.user_interaction.display_info("[STEP 3/3] Model ready for inference!")
            
            # Log model info
            total_params = sum(p.numel() for p in self._model.parameters())
            self.logger.info(f"MiniMax model initialized: {total_params:,} parameters")
            
        except Exception as e:
            error_msg = f"Failed to initialize MiniMax model: {e}"
            self.logger.error(error_msg)
            raise ModelConfigurationError(error_msg)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate text from a prompt using MiniMax-M2.1.

        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters (skip_prompt, max_new_tokens)

        Returns:
            GenerationResult with content and metadata

        Raises:
            GenerationError: If generation fails
        """
        try:
            # Check if user wants to continue
            skip_prompt = kwargs.get('skip_prompt', False)
            if not skip_prompt and not self.user_interaction.prompt_continue():
                self.user_interaction.display_info("Generation skipped by user.")
                return GenerationResult(content="Generation skipped by user.")

            self.user_interaction.display_info(f"\nPrompt: {prompt[:100]}...")
            self.user_interaction.display_info("[GENERATING] Processing your request...")

            # Prepare messages in chat format (as per MiniMax documentation)
            messages = [
                {"role": "user", "content": prompt},
            ]

            # Apply chat template
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device if hasattr(self._model, 'device') else self._device)

            # Count prompt tokens
            prompt_tokens = inputs["input_ids"].shape[-1]
            self.logger.info(f"Prompt tokens: {prompt_tokens}")
            
            # Log token usage for input
            self.token_manager.log_usage(
                self.config.model_name,
                prompt_tokens,
                "prompt",
                is_output=False
            )

            # Generate text
            max_new_tokens = kwargs.get('max_new_tokens', self.config.max_tokens)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True
            )

            # Decode the response (skip the input tokens)
            response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            generated_text = self._tokenizer.decode(response_tokens)

            self.logger.info(f"Generated {len(response_tokens)} tokens")
            self.user_interaction.display_info("[SUCCESS] Response generated!")

            # Log token usage for output
            self.token_manager.log_usage(
                self.config.model_name,
                len(response_tokens),
                "response",
                is_output=True
            )

            # Get usage summary
            usage_summary = self.token_manager.get_usage_summary()

            return GenerationResult(
                content=generated_text,
                tokens_used=len(response_tokens),
                cost=usage_summary.get('total_cost', 0.0),
                metadata={
                    "provider": self.provider,
                    "model": self.config.model_name,
                    "device": self._device,
                    "dtype": str(self._dtype),
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": len(response_tokens),
                    "total_tokens": prompt_tokens + len(response_tokens),
                }
            )

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            self.logger.error(error_msg)
            self.user_interaction.display_error(error_msg)
            raise GenerationError(error_msg)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        if self._model is None:
            return {
                "provider": self.provider,
                "model_name": self.config.model_name,
                "status": "not_initialized"
            }

        total_params = sum(p.numel() for p in self._model.parameters())
        return {
            "provider": self.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "device": self._device,
            "dtype": str(self._dtype),
            "parameters": total_params,
            "status": "ready"
        }

    @property
    def provider(self) -> str:
        """Get the model provider name."""
        return "MiniMax"

    def cleanup(self) -> None:
        """Clean up model resources to free memory."""
        try:
            self.logger.info("Cleaning up MiniMax model resources...")
            
            if self._model is not None:
                del self._model
                self._model = None
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear device-specific caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            self.logger.info("Model resources cleaned up successfully")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
