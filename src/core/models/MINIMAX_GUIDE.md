# MiniMax-M2.1 Integration Guide

Complete guide for using MiniMax-M2.1 and other HuggingFace models in your LangChain project.

## Overview

MiniMax-M2.1 is a powerful open-source language model available on HuggingFace. It can be used locally without API keys, making it ideal for:
- Running models on your hardware
- Avoiding API costs
- Complete data privacy
- Fine-grained control over inference

## Quick Start

### Installation

```bash
# Core dependencies
pip install transformers torch

# Optional: For faster inference
pip install accelerate
```

### Basic Usage

#### Option 1: Using the Factory (Recommended)

```python
from src.core.dependency_injection import get_container

container = get_container()
factory = container.resolve('model_factory')

# Create MiniMax model
model = factory.create_minimax_model()

# Generate text
result = model.generate("Explain AI in simple terms")
print(result.content)
```

#### Option 2: Using the Factory with Custom Config

```python
from src.core.interfaces import ModelConfig

config = ModelConfig(
    model_name="MiniMaxAI/MiniMax-M2.1",
    temperature=1.0,  # MiniMax recommends 1.0
    max_tokens=512
)

model = factory.create_model("minimax", config)
result = model.generate("Your prompt here")
```

#### Option 3: Using Any HuggingFace Model

```python
# Works with any HuggingFace model supporting transformers
model = factory.create_huggingface_model(
    model_name="mistralai/Mistral-7B-v0.1",
    temperature=0.7,
    max_tokens=512
)
```

## MiniMax-M2.1 Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 229B |
| **License** | Modified MIT |
| **Quantization** | Supports FP8, BF16, FP32 |
| **Input Format** | Chat messages (with `apply_chat_template`) |
| **Recommended Temperature** | 1.0 |
| **Recommended top_p** | 0.95 |
| **Recommended top_k** | 40 |

## Memory Requirements

| Configuration | VRAM Needed | Hardware |
|--------------|-----------|----------|
| Full FP32 | ~450GB | 8x H100 |
| BF16 | ~225GB | 2-4x H100 |
| FP8 | ~60GB | Single A100 |
| Quantized GGUF | ~30-60GB | Consumer GPU |

**Recommendation**: Use quantized versions for consumer hardware. Check:
https://huggingface.co/models?other=base_model:quantized:MiniMaxAI/MiniMax-M2.1

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│   Your Application                   │
│   (any code using ILanguageModel)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   ModelFactory                       │
│   - Creates model instances          │
│   - Routes to correct provider       │
│   - Handles dependencies             │
└──────┬───────────────────────┬──────┘
       │                       │
       ▼                       ▼
┌────────────────┐    ┌───────────────────┐
│  ClaudeModel   │    │  MiniMaxModel     │
│  (API-based)   │    │  (Local, free)    │
└────────────────┘    └───────┬───────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Transformers     │
                    │  (HuggingFace)    │
                    └───────────────────┘
```

### Dependency Injection

All models receive the same dependencies:

```python
class MiniMaxModel(ILanguageModel):
    def __init__(
        self,
        config: ModelConfig,           # Model configuration
        token_manager: ITokenManager,  # Token counting & cost tracking
        user_interaction: IUserInteraction,  # User prompts & display
        logger: logging.Logger,        # Logging
        cost_tracker: Optional[Any] = None  # Cost tracking
    ):
```

## Interchangeable Usage

All models implement `ILanguageModel`, so you can write provider-agnostic code:

```python
def my_ai_function(model: ILanguageModel, prompt: str) -> str:
    """Works with any model provider."""
    result = model.generate(prompt)
    return result.content

# Use with Claude (API)
claude = factory.create_default_claude_model()
output = my_ai_function(claude, "Hello")

# Use with MiniMax (Local)
minimax = factory.create_minimax_model()
output = my_ai_function(minimax, "Hello")

# Both work identically!
```

## Configuration

### Model Configuration Options

```python
from src.core.interfaces import ModelConfig, GenerationStrategy

config = ModelConfig(
    model_name="MiniMaxAI/MiniMax-M2.1",
    temperature=1.0,
    max_tokens=512,
    system_message="You are a helpful assistant",
    generation_strategy=GenerationStrategy.STANDARD,
    additional_params={
        "top_p": 0.95,
        "top_k": 40
    }
)

model = factory.create_model("minimax", config)
```

### Environment Variables

Create `.env` in your project root:

```bash
# Optional: HuggingFace token (for downloading private models or avoiding rate limits)
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxx

# Optional: Anthropic token (for Claude API)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

## Generation Process

### Step-by-Step Generation

```python
result = model.generate(
    prompt="What is quantum computing?",
    skip_prompt=False  # Ask user confirmation before generating
)

# Result contains:
print(result.content)          # Generated text
print(result.tokens_used)      # Token count
print(result.cost)             # Estimated cost
print(result.metadata)         # Additional info
```

### Generation Flow

```
1. Validate prompt
2. Ask user confirmation (if needed)
3. Count input tokens
4. Format input for chat template
5. Generate output tokens
6. Decode response
7. Log usage (for analytics)
8. Return GenerationResult with metadata
```

## Comparing Models

### Claude vs MiniMax

| Feature | Claude | MiniMax |
|---------|--------|---------|
| **Cost** | API pricing | Free (local) |
| **Privacy** | Data sent to API | All local |
| **Speed** | Network latency | Direct inference |
| **Quality** | Very high | Very high |
| **Customization** | Limited | Full control |
| **Requires** | API key | GPU/CPU |
| **Best for** | Production APIs | Local apps |

## Advanced Usage

### Custom Model Registration

```python
from src.core.models import ModelFactory

# Register your custom model
factory.register_model("mymodel", MyCustomModel)

# Use it
model = factory.create_model("mymodel", config)
```

### Token Counting

```python
container = get_container()
token_manager = container.resolve('token_manager')

tokens = token_manager.count_tokens("Your text", "MiniMaxAI/MiniMax-M2.1")
print(f"Token count: {tokens}")
```

### Cost Tracking

```python
# Get detailed usage statistics
summary = token_manager.get_usage_summary()
print(f"Total tokens: {summary['total_tokens']}")
print(f"Total cost: ${summary['total_cost']}")
```

## Troubleshooting

### `OutOfMemoryError`

**Problem**: GPU ran out of memory

**Solutions**:
1. Use a quantized version: `hf.co/models?other=base_model:quantized:MiniMaxAI/MiniMax-M2.1`
2. Reduce `max_tokens`
3. Use `device_map="cpu"` (slower but uses system RAM)
4. Use a smaller model instead

### `trust_remote_code=True` Warning

**Why it exists**: The model uses custom code from HuggingFace

**Why it's safe**: MiniMax-M2.1 is from a trusted organization. The code is:
- Visible on HuggingFace
- Audited by the community
- Necessary for proper inference

### Model Download Issues

**Problem**: Model fails to download

**Solutions**:
1. Set HuggingFace token: `HUGGINGFACE_API_KEY=...`
2. Check disk space (need ~450GB for full model)
3. Verify internet connection
4. Try: `huggingface-cli login`

### Device Not Found

**Problem**: CUDA/GPU not detected

**Solutions**:
```python
import torch
print(torch.cuda.is_available())  # Check if CUDA available
print(torch.cuda.get_device_name(0))  # Get GPU name
print(torch.cuda.get_device_properties(0))  # Get specs
```

## Performance Tips

1. **Batch Processing**: Process multiple prompts at once
2. **GPU Optimization**: Use `device_map="auto"` for multi-GPU
3. **BF16 Precision**: Use bfloat16 for better speed/quality balance
4. **Quantization**: Use Q4 or FP8 for consumer hardware
5. **Context Reuse**: Reuse tokenizer and model instances

## API Reference

### MiniMaxModel Class

```python
class MiniMaxModel(ILanguageModel):
    def __init__(config, token_manager, user_interaction, logger)
    
    def generate(prompt: str, **kwargs) -> GenerationResult
    def get_model_info() -> Dict[str, Any]
    def cleanup() -> None
    
    @property
    def provider() -> str  # Returns "MiniMax"
```

### ModelFactory Methods

```python
# Create with default settings
model = factory.create_minimax_model()

# Create with custom settings
model = factory.create_minimax_model(
    model_name="MiniMaxAI/MiniMax-M2.1",
    temperature=1.0,
    max_tokens=512
)

# Create any HuggingFace model
model = factory.create_huggingface_model(
    model_name="huggingface_model_id",
    temperature=0.7,
    max_tokens=512
)

# Get available providers
providers = factory.get_available_providers()
# Returns: ["anthropic", "minimax", "huggingface"]
```

## Examples

See `examples_minimax.py` in the project root for:
- Basic usage
- Factory methods
- Custom configurations
- Interchangeable models
- Provider comparison

Run examples:
```bash
python examples_minimax.py
```

## Resources

- **MiniMax Website**: https://www.minimax.io/
- **Model Card**: https://huggingface.co/MiniMaxAI/MiniMax-M2.1
- **GitHub**: https://github.com/MiniMax-AI/MiniMax-M2.1
- **Paper**: https://arxiv.org/abs/2509.06501
- **Transformers Docs**: https://huggingface.co/docs/transformers

## Support

For issues or questions:
- MiniMax Discord: https://discord.com/invite/hvvt8hAye6
- GitHub Discussions: https://github.com/MiniMax-AI/MiniMax-M2.1/discussions
- This Project Issues: Check the repository

## License

- MiniMax-M2.1: Modified MIT
- This integration: Follow your project license
