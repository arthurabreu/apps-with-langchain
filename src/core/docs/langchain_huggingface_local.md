# langchain_huggingface_local.py

**📍 Optional reference (experimental)** | [← Back to Index](index.md) | Read when: exploring local model support

> Educational module for running local Hugging Face models with LangChain. Demonstrates device detection, model loading, and memory management.

## What This File Does

This module lets you run open-source language models **locally** (no API calls, no cost). Useful for:

- Learning how transformers work
- Testing without API rate limits
- Privacy (no data sent to cloud)
- Cost-free experimentation

Key features:
- Auto-detect GPU (CUDA, MPS) or fall back to CPU
- Load models from Hugging Face Hub
- Run text generation with proper device memory management
- Display loading progress with animated spinner
- Token counting using tokenizer
- Proper cleanup (free GPU/CPU memory after use)

**Note:** This is **experimental/educational**. Current model registry maps to a Kotlin model which may not be ideal for all tasks.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `_select_device()` | function | Detect best device (CUDA/MPS/CPU) and dtype |
| `LoadingSpinner` | class | Animated loading indicator |
| `.start(message)` | method | Start spinner animation |
| `.stop(final_message)` | method | Stop spinner, clean up |
| `.spin()` | method | Animation loop (runs in thread) |
| `LocalHuggingFaceModel` | class | Main class for running local models |
| `.__init__(model_id, device, ...)` | method | Initialize model (don't load yet) |
| `._initialize_model()` | method | Load tokenizer, model, pipeline (lazy) |
| `.generate(prompt, skip_prompt, **kwargs)` | method | Generate text from prompt |
| `._format_prompt(prompt)` | method | Prepare prompt for model |
| `.cleanup()` | method | Free GPU/CPU memory |
| `.__enter__() / __exit__()` | methods | Context manager support |
| `.get_model_info()` | method | Return model device/dtype info |
| `.test_connection()` | method | Quick test generation |

---

## _select_device() Function

```python
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
        dtype = torch.float16  # Half precision on GPU = faster + less VRAM
        ...
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        dtype = torch.float16
        ...
    else:
        device = "cpu"
        dtype = torch.float32  # Full precision on CPU (float16 would be too inaccurate)
        ...
    return device, dtype
```

**Module-level function** (not a method) because it's a stateless utility. Call it once at model load time to choose device.

**Kotlin equivalent:**
```kotlin
fun selectDevice(): Pair<String, TorchDtype> {
    return when {
        cuda.isAvailable() -> "cuda" to TorchDtype.FLOAT16
        torch.backends.mps.isAvailable() -> "mps" to TorchDtype.FLOAT16
        else -> "cpu" to TorchDtype.FLOAT32
    }
}
```

---

## LoadingSpinner Class

Displays a rotating spinner while loading models (they can take minutes).

```python
class LoadingSpinner:
    def __init__(self, message: str = "Loading", delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.spinner_chars = ['⠋', '⠙', '⠹', ...]  # Braille animation
        self.busy = False
        self.spinner_thread = None

    def start(self, message: Optional[str] = None):
        """Start spinner in background thread."""
        self.busy = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True  # Dies when main thread exits
        self.spinner_thread.start()

    def stop(self, final_message: str = ""):
        """Stop spinner and clear line."""
        self.busy = False
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)
        # Clear spinner line, print final message
```

**Usage:**
```python
spinner = LoadingSpinner()
spinner.start("Loading tokenizer")
# ... do work ...
spinner.stop("[SUCCESS] Tokenizer loaded")
```

**Key:** `daemon=True` thread dies automatically when main thread exits (won't block shutdown).

**Kotlin equivalent:**
```kotlin
thread(isDaemon = true) {
    while (busy) {
        print("\r${message} ${spinner_chars[index]}")
        Thread.sleep(delay.toLong())
    }
}
```

---

## LocalHuggingFaceModel Class

Main class for local model inference.

### __init__

```python
def __init__(
    self,
    model_id: Optional[str] = None,
    device: str = "auto",
    max_length: int = 512,
    temperature: float = 0.2
):
```

**Note:** Constructor **does not load the model**. That happens in `_initialize_model()` (lazy initialization). This lets you create the object quickly and load only when needed.

**Parameters:**
- `model_id` — HF Hub model ID like "JetBrains/Mellum-4b-sft-kotlin"
- `device` — "auto" (auto-detect), "cuda", "mps", or "cpu"
- `max_length` — max tokens to generate
- `temperature` — 0.0 (deterministic) to 1.0 (random)

### _initialize_model() (Lazy)

```python
def _initialize_model(self):
    """Initialize the model, tokenizer, and pipeline."""
    if self._pipeline is not None:
        return self._pipeline  # Already loaded, return existing

    # Step 1: Load tokenizer
    self._tokenizer = AutoTokenizer.from_pretrained(
        self.model_id,
        token=os.getenv("HUGGINGFACE_API_KEY")
    )

    # Step 2: Load model (device-specific)
    device, dtype = _select_device()
    if device == "cuda":
        self._model = AutoModelForCausalLM.from_pretrained(
            ...,
            device_map="auto",  # Distribute across GPUs if multiple
            torch_dtype=dtype
        )
    elif device == "mps":
        self._model = AutoModelForCausalLM.from_pretrained(...)
        self._model = self._model.to(device)  # Move to MPS
    else:
        # CPU: load normally
        self._model = AutoModelForCausalLM.from_pretrained(...)

    # Step 3: Create pipeline
    self._pipeline = pipeline("text-generation", model=..., tokenizer=...)
```

**Pattern:** Check if already initialized (`if self._pipeline is None`), only load once. This is **lazy initialization** — defer work until needed.

**Kotlin equivalent:** `by lazy { initialize() }` delegate

---

### generate()

```python
def generate(self, prompt: str, skip_prompt: bool = False, **kwargs) -> Optional[str]:
    """Generate text from a prompt."""
    if self._pipeline is None:
        self._initialize_model()  # Lazy load on first use

    # Format prompt for model
    formatted_prompt = self._format_prompt(prompt)

    # Token analysis
    prompt_tokens = len(self._tokenizer.encode(formatted_prompt))

    # Generate
    result = self._pipeline(
        formatted_prompt,
        max_new_tokens=self.max_length,
        temperature=self.temperature,
        do_sample=True
    )

    # Extract and clean response
    generated_text = result[0]['generated_text']
    response = generated_text[len(formatted_prompt):].strip()  # Remove input

    return response
```

**Returns:** Generated text or None if user skipped.

---

### cleanup()

```python
def cleanup(self):
    """
    Explicitly free GPU/CPU memory.
    Important: Python's garbage collector doesn't auto-free GPU memory.
    """
    del self._pipeline
    del self._model
    del self._tokenizer

    import gc
    gc.collect()  # Force garbage collection

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Free GPU memory
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()  # Free MPS memory
```

**Why explicit cleanup?** Python's garbage collector may not free GPU memory immediately. Calling `cleanup()` ensures you free resources before exiting.

---

### Context Manager Support

```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.cleanup()
```

**Usage:**
```python
with LocalHuggingFaceModel(model_id="...") as model:
    result = model.generate("Hello")
    print(result)
# Automatically calls cleanup() on exit
```

**Kotlin equivalent:**
```kotlin
class LocalHuggingFaceModel : Closeable {
    override fun close() = cleanup()
}
// Use with .use {}
model.use { m -> m.generate("Hello") }
```

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `Tuple[str, torch.dtype]` | `Pair<String, TorchDtype>` | _select_device return |
| `torch.cuda.is_available()` | CUDA availability check in Kotlin | _select_device |
| `threading.Thread(target=func)` | `thread { func() }` | LoadingSpinner |
| `daemon=True` | `isDaemon = true` | LoadingSpinner thread |
| `Optional[str]` | `String?` | method params |
| `if self._pipeline is None` | `if (_pipeline == null)` | Lazy init pattern |
| `AutoTokenizer.from_pretrained(...)` | HF Java/Kotlin library equivalent | _initialize_model |
| `**kwargs` | Kotlin named params or `vararg` | generate method |
| `del obj` | No equivalent (auto GC); manual cleanup | cleanup() |
| `gc.collect()` | `System.gc()` | cleanup() |
| `__enter__` / `__exit__` | `Closeable.use {}` | Context manager |
| `os.getenv("KEY")` | `System.getenv("KEY")` | Loading from env |

---

## How It Connects to Other Files

**Imports from:**
- Standard library (`os`, `threading`, `time`, `sys`, `typing`, `gc`)
- External: `torch`, `transformers` (HuggingFace), `langchain_core`

**Imported by:**
- `model_comparison.py` — optional import for model comparison
- Examples/demos

**Flow:**
1. Code creates `LocalHuggingFaceModel(model_id="...")`
2. Constructor returns immediately; model not loaded yet
3. Code calls `.generate(prompt)`
4. `.generate()` calls `._initialize_model()` if not already loaded
5. First generation may take minutes (loads tokenizer, model, pipeline)
6. Subsequent generations are fast (model cached)
7. Code calls `.cleanup()` or uses context manager
8. Memory is freed, app can continue or exit

**Note:** This is an **optional/educational** module. The main app uses `ClaudeModel` and strategies. This module exists for learning and comparison purposes.

---

## Key Patterns

**Lazy initialization:**
```python
if self._pipeline is None:
    self._initialize_model()
# Only initialize once, on first use
```

**Device selection:**
```python
device, dtype = _select_device()
if device == "cuda":
    # CUDA-specific loading
elif device == "mps":
    # Apple Silicon-specific loading
else:
    # CPU loading
```

**Context manager (automatic cleanup):**
```python
with model as m:
    m.generate(...)
# cleanup() called automatically
```

**Spinner in background thread:**
```python
spinner = LoadingSpinner()
spinner.start("Loading...")
# Do slow work
spinner.stop("[SUCCESS] Done!")
# Spinner runs in background, doesn't block main thread
```

---

**[← Back to Index](index.md) | [Back to core learning path →](strategies/streaming_generation.md)**

*Reference doc: explore when you want to add local model support*
