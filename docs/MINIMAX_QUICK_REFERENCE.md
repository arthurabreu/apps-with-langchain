# MiniMax Quick Reference

## One-Liner to Get Started

```python
model = get_container().resolve('model_factory').create_minimax_model()
result = model.generate("Your prompt here")
print(result.content)
```

## Installation

Model dependencies are already in `requirements.txt`:
- ✅ `torch>=2.0.0`
- ✅ `transformers>=4.30.0`  
- ✅ `accelerate>=0.20.0`

No extra `pip install` needed!

## Usage Patterns

### Pattern 1: Simplest
```python
from src.core.dependency_injection import get_container

factory = get_container().resolve('model_factory')
model = factory.create_minimax_model()
result = model.generate("Your prompt")
print(result.content)
```

### Pattern 2: Custom Config
```python
from src.core.interfaces import ModelConfig

config = ModelConfig(
    model_name="MiniMaxAI/MiniMax-M2.1",
    temperature=1.0,
    max_tokens=512
)
model = factory.create_model("minimax", config)
result = model.generate("Your prompt")
```

### Pattern 3: Any HuggingFace Model
```python
model = factory.create_huggingface_model(
    "mistralai/Mistral-7B-v0.1",
    temperature=0.7
)
result = model.generate("prompt")
```

### Pattern 4: Interchangeable (Works with ANY model)
```python
def process_text(model: ILanguageModel, text: str) -> str:
    result = model.generate(text)
    return result.content

# Use with Claude
claude = factory.create_default_claude_model()
output = process_text(claude, "prompt")

# Use with MiniMax
minimax = factory.create_minimax_model()
output = process_text(minimax, "prompt")
# Same code, different provider!
```

## Common Tasks

### Get Model Info
```python
info = model.get_model_info()
# Returns: {'provider': 'MiniMax', 'model_name': '...', 'status': 'ready', ...}
```

### Count Tokens
```python
token_manager = get_container().resolve('token_manager')
tokens = token_manager.count_tokens(text, "MiniMaxAI/MiniMax-M2.1")
```

### Get Usage Stats
```python
summary = token_manager.get_usage_summary()
print(f"Total tokens: {summary['total_tokens']}")
print(f"Total cost: ${summary['total_cost']}")
```

### Custom Generation Parameters
```python
result = model.generate(
    "prompt",
    max_new_tokens=1024,
    skip_prompt=True  # Don't ask for confirmation
)
```

### Clean Up Resources
```python
model.cleanup()  # Free GPU memory
```

## Recommended Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| temperature | 1.0 | MiniMax docs recommend |
| top_p | 0.95 | MiniMax default |
| top_k | 40 | MiniMax default |
| max_tokens | 512 | Reasonable default |

## List Available Providers

```python
providers = factory.get_available_providers()
print(providers)  # ['anthropic', 'minimax', 'huggingface']
```

## Environment Setup

Create `.env` (optional):
```bash
HUGGINGFACE_API_KEY=hf_xxxxx  # For private models
ANTHROPIC_API_KEY=sk-ant-xxx  # For Claude (still supported)
```

## Common Issues

| Issue | Solution |
|-------|----------|
| `OutOfMemoryError` | Use quantized model variant |
| Model download slow | Set `HUGGINGFACE_API_KEY` to get priority |
| CUDA not detected | Check: `torch.cuda.is_available()` |

## Result Object

```python
result = model.generate("prompt")

result.content          # Generated text (str)
result.tokens_used      # Number of tokens (int)
result.cost            # Estimated cost (float, 0 for local)
result.metadata        # Extra info (dict)
```

## Switch Between Models

```python
# Just change one line!
# model = factory.create_default_claude_model()
model = factory.create_minimax_model()

# Rest of code works exactly the same
result = model.generate(prompt)
```

## File Locations

| File | Purpose |
|------|---------|
| [src/core/models/minimax_model.py](src/core/models/minimax_model.py) | Implementation |
| [src/core/models/MINIMAX_GUIDE.md](src/core/models/MINIMAX_GUIDE.md) | Full guide |
| [examples_minimax.py](examples_minimax.py) | Working examples |
| [MINIMAX_IMPLEMENTATION.md](MINIMAX_IMPLEMENTATION.md) | What was implemented |

## Performance Tips

1. **Batch processing**: Generate multiple responses in a loop
2. **Reuse objects**: Don't recreate model for each request
3. **BF16 precision**: Use BF16 for speed/quality balance
4. **GPU acceleration**: Always use CUDA if available
5. **Quantized models**: Use GGUF format for consumer hardware

## Memory Quick Guide

```
MiniMax-M2.1:
- Full model (FP32):      ~450GB
- Half precision (FP16):  ~225GB
- Brain float (BF16):     ~225GB
- 8-bit (FP8):            ~60GB
- Quantized (Q4_K_M):     ~30GB

Recommendation: For consumer GPUs, use Q4 GGUF versions
```

## Testing

```python
# Quick test
from src.core.models import MiniMaxModel
print("✓ MiniMax available!")

# Full factory test
factory = get_container().resolve('model_factory')
model = factory.create_minimax_model()
print(f"✓ Model created: {model.provider}")
```

## Further Reading

- **[MINIMAX_GUIDE.md](src/core/models/MINIMAX_GUIDE.md)** - Comprehensive guide
- **[examples_minimax.py](examples_minimax.py)** - 5 working examples
- **[Official MiniMax Docs](https://github.com/MiniMax-AI/MiniMax-M2.1)** - Model documentation

---

**TL;DR**: Call `factory.create_minimax_model()` and use it like any other language model!
