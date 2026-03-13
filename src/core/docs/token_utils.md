# token_utils.md

> Token counting, cost estimation, and usage tracking for LLM operations. Monitors consumption and budget.

## What This File Does

Managing LLM costs is crucial. This file provides:

1. **TokenUsage** (dataclass) — records a single usage event (timestamp, model, tokens, cost, operation type)
2. **TokenManager** — class with:
   - Class-level pricing tables (cost per 1M tokens by model)
   - Context window limits
   - Model aliases ("claude-3-haiku" → "claude-3-haiku-20240307")
   - Methods to count, estimate cost, log usage, compare models

**Pricing note:** Prices are **per 1M tokens** and may change. Prices in code are from 2026-01 and should be verified against official provider docs.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `TokenUsage` | @dataclass | Records one usage event (timestamp, model, tokens, cost, op_type) |
| `TokenManager` | class | Main token management service |
| `.resolve_model(model)` | @classmethod | Map alias to canonical model name |
| `.count_tokens(text, model)` | method | Estimate token count for text |
| `.estimate_cost(num_tokens, model, is_output)` | method | Convert tokens to USD |
| `.get_token_breakdown(text, model)` | method | Detailed breakdown of tokens & cost |
| `.check_context_window(text, model)` | method | Check if tokens fit in context; return remaining |
| `.log_usage(model, tokens, operation_type, is_output)` | method | Record usage to history & file |
| `.get_coding_model(use_case)` | method | Get best model for coding task |
| `.list_all_models()` | method | Get all supported models |
| `.get_model_info(model)` | method | Detailed pricing/spec for one model |
| `.compare_models(models, tokens_count)` | method | Compare costs across models |
| `.get_usage_summary()` | method | Aggregate usage stats |
| `._load_usage_history()` | method | Load usage from JSON file |
| `._save_usage_history()` | method | Save usage to JSON file |
| `main()` | function | Example usage when run as script |

---

## TokenUsage (@dataclass)

Records a single token usage event.

```python
@dataclass
class TokenUsage:
    timestamp: str             # ISO format: "2026-03-13T14:25:30.123456"
    model: str                 # e.g., "claude-3-haiku-20240307"
    tokens_used: int           # Total tokens in operation
    estimated_cost: float      # USD cost of this operation
    operation_type: str        # "chat", "completion", "hf_prompt", "hf_response", etc.
```

**Example:**
```python
usage = TokenUsage(
    timestamp="2026-03-13T14:25:30.123456",
    model="claude-3-haiku-20240307",
    tokens_used=150,
    estimated_cost=0.000375,
    operation_type="standard_prompt"
)
```

**Kotlin equivalent:**
```kotlin
data class TokenUsage(
    val timestamp: String,
    val model: String,
    val tokensUsed: Int,
    val estimatedCost: Float,
    val operationType: String
)
```

---

## TokenManager (Detailed)

### Class-Level Constants

**COST_PER_1M_TOKENS** — Pricing table:
```python
{
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, ...},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00, ...},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, ...},
}
```
Output tokens cost 5-300x more than input tokens (larger models cost more).

**MAX_TOKENS** — Context window size for each model:
```python
{
    "claude-3-opus-20240229": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-haiku-20240307": 200000,
}
```

**MODEL_ALIASES** — Friendly names mapped to canonical versions:
```python
{
    "claude-3-haiku": {"model": "claude-3-haiku-20240307", "description": "..."},
    ...
}
```

**BEST_CODING_MODELS** — Recommended models by use case:
```python
{
    "general-purpose": "claude-3-5-sonnet-20240620",
    "fast-coding": "claude-3-haiku-20240307",
    "advanced-logic": "claude-3-opus-20240229",
}
```

### __init__

```python
def __init__(self, log_file: str = "token_usage.json"):
    self.log_file = log_file
    self.usage_history: List[TokenUsage] = self._load_usage_history()
```

**Usage:**
```python
tm = TokenManager()  # Loads existing token_usage.json if present
tm = TokenManager("my_custom.json")  # Use different log file
```

### resolve_model (@classmethod)

```python
@classmethod
def resolve_model(cls, model: str) -> str:
    """Map aliases to canonical model names."""
    if model in cls.MODEL_ALIASES:
        return cls.MODEL_ALIASES[model]["model"]
    return model  # Return unchanged if not an alias
```

**Example:**
```python
TokenManager.resolve_model("claude-3-haiku")  # → "claude-3-haiku-20240307"
TokenManager.resolve_model("claude-3-haiku-20240307")  # → "claude-3-haiku-20240307"
TokenManager.resolve_model("unknown-model")  # → "unknown-model"
```

**Kotlin equivalent:** `companion object fun resolveModel(model: String): String`

### count_tokens()

```python
def count_tokens(self, text: str, model: str = "claude-3-haiku-20240307") -> int:
    """Estimate token count (rough: ~1 token per 4 chars for English)."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return int(len(text) / 4)
```

**Note:** This is an approximation. Claude's actual tokenizer is not open-sourced, so we estimate. For production, consider using LangChain's `ChatAnthropic.get_token_ids()` or Anthropic's official token counter if available.

**Example:**
```python
tm = TokenManager()
count = tm.count_tokens("Hello, world!", "claude-3-haiku-20240307")
# "Hello, world!" = 13 chars → ~3 tokens
```

### estimate_cost()

```python
def estimate_cost(self, num_tokens: int, model: str, is_output: bool = False) -> float:
    """Convert token count to USD."""
    model = self.resolve_model(model)
    if model not in self.COST_PER_1M_TOKENS:
        return 0.0

    rate_type = "output" if is_output else "input"
    rate = self.COST_PER_1M_TOKENS[model][rate_type]
    return (num_tokens / 1_000_000) * rate
```

**Formula:**
```
cost = (tokens / 1,000,000) * rate_per_1m
```

**Example:**
```python
tm = TokenManager()

# 1000 input tokens on claude-3-haiku
cost = tm.estimate_cost(1000, "claude-3-haiku", is_output=False)
# (1000 / 1,000,000) * 0.25 = 0.00000025 USD

# 500 output tokens on claude-3-5-sonnet
cost = tm.estimate_cost(500, "claude-3-5-sonnet", is_output=True)
# (500 / 1,000,000) * 15.00 = 0.0000075 USD
```

### get_token_breakdown()

```python
def get_token_breakdown(self, text: str, model: str) -> Dict:
    """Get detailed breakdown: total tokens, chars, tokens/char, cost."""
    return {
        "total_tokens": token_count,
        "total_chars": len(text),
        "tokens_per_char": ...,
        "estimated_cost": ...,
        "model": model,
    }
```

**Example:**
```python
tm = TokenManager()
breakdown = tm.get_token_breakdown("Long text here...", "claude-3-haiku")
print(breakdown)
# {
#     "total_tokens": 5,
#     "total_chars": 17,
#     "tokens_per_char": 0.294,
#     "estimated_cost": 0.00000125,
#     "model": "claude-3-haiku-20240307"
# }
```

### check_context_window()

```python
def check_context_window(self, text: str, model: str) -> Tuple[int, int, float]:
    """Check if tokens fit; return (token_count, remaining, usage_percent)."""
    token_count = self.count_tokens(text, model)
    max_tokens = self.MAX_TOKENS.get(model_resolved, 0)
    remaining = max_tokens - token_count if max_tokens > 0 else 0
    usage_percentage = (token_count / max_tokens * 100) if max_tokens > 0 else 0

    return token_count, remaining, usage_percentage
```

**Example:**
```python
tm = TokenManager()
tokens, remaining, percent = tm.check_context_window("text", "claude-3-haiku")
# tokens = 1, remaining = 199999, percent = 0.0005
if percent > 80:
    print("Warning: 80%+ of context used")
```

**Kotlin equivalent:** `fun checkContextWindow(...): Triple<Int, Int, Float>`

### log_usage()

```python
def log_usage(self, model: str, tokens_used: int, operation_type: str, is_output: bool = False):
    """Record usage event to history & file."""
    usage = TokenUsage(
        timestamp=datetime.now().isoformat(),
        model=model_resolved,
        tokens_used=tokens_used,
        estimated_cost=cost,
        operation_type=operation_type
    )
    self.usage_history.append(usage)
    self._save_usage_history()  # Persist to JSON
```

**Example:**
```python
tm = TokenManager()
tm.log_usage("claude-3-haiku", 100, "standard_prompt", is_output=False)
tm.log_usage("claude-3-haiku", 50, "standard_response", is_output=True)
# Appends to usage_history, saves to token_usage.json
```

### get_usage_summary()

```python
def get_usage_summary(self) -> Dict:
    """Aggregate usage by model and operation type."""
    return {
        "total_tokens": ...,
        "total_cost": ...,
        "usage_by_model": {
            "claude-3-haiku": {"tokens": 150, "cost": 0.000375},
            ...
        },
        "usage_by_operation": {
            "standard_prompt": {"tokens": 100, "cost": 0.000250},
            ...
        }
    }
```

**Example:**
```python
tm = TokenManager()
summary = tm.get_usage_summary()
print(f"Total cost so far: ${summary['total_cost']:.6f}")
for model, usage in summary['usage_by_model'].items():
    print(f"{model}: {usage['tokens']} tokens (${usage['cost']:.6f})")
```

### get_model_info()

```python
def get_model_info(self, model: str) -> Dict:
    """Get pricing, context window, description for a model."""
    return {
        "model": model,
        "input_cost_per_1m": 0.25,
        "output_cost_per_1m": 1.25,
        "description": "Fastest and most affordable Claude",
        "category": "efficient",
        "context_window": 200000
    }
```

**Example:**
```python
tm = TokenManager()
info = tm.get_model_info("claude-3-haiku")
print(f"{info['description']} - ${info['input_cost_per_1m']}/1M input tokens")
```

### compare_models()

```python
def compare_models(self, models: List[str], tokens_count: int = 1000) -> Dict:
    """Compare cost of same tokens across models."""
    return {
        "claude-3-haiku": {
            "resolved_model": "claude-3-haiku-20240307",
            "input_cost": 0.00000025,
            "output_cost": 0.00000125,
            "total_estimated": 0.0000015,
            "description": "Fastest and most affordable Claude"
        },
        ...
    }
```

**Example:**
```python
tm = TokenManager()
comparison = tm.compare_models(
    ["claude-3-haiku", "claude-3-5-sonnet", "claude-3-opus"],
    tokens_count=1000
)
for model, costs in comparison.items():
    print(f"{model}: ${costs['total_estimated']:.6f} for 1000 tokens")
```

### get_coding_model()

```python
def get_coding_model(self, use_case: str = "general-purpose") -> str:
    """Get best coding model for use case."""
    # "general-purpose" → sonnet (balanced)
    # "fast-coding" → haiku (quick)
    # "advanced-logic" → opus (powerful)
```

**Example:**
```python
tm = TokenManager()
model = tm.get_coding_model("fast-coding")  # → "claude-3-haiku-20240307"
```

### list_all_models()

```python
def list_all_models(self) -> List[str]:
    """Get list of all supported models."""
    return sorted(list(self.COST_PER_1M_TOKENS.keys()))
```

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `@dataclass` | `data class` | TokenUsage |
| `@classmethod` | `companion object fun` | resolve_model |
| `Dict[str, Dict[str, float]]` | `Map<String, Map<String, Float>>` | COST_PER_1M_TOKENS |
| `List[TokenUsage]` | `List<TokenUsage>` | usage_history |
| `Optional[str]` | `String?` | type hints |
| `Tuple[int, int, float]` | `Triple<Int, Int, Float>` | return type of check_context_window |
| `datetime.now().isoformat()` | `Instant.now().toString()` or Java interop | timestamp field |
| `[item for item in list]` | `list.map { ... }` | list comprehensions |
| `**dict` unpacking | Named parameters or `mapToParams()` | TokenUsage(**dict) |
| `if __name__ == "__main__"` | `fun main(args: Array<String>)` | Module guard at bottom |
| `json.dump([vars(obj) for ...]` | `Gson().toJson(list)` | _save_usage_history |
| `json.load()` returning dicts | `Gson().fromJson()` | _load_usage_history |

---

## How It Connects to Other Files

**Imports from:** Standard library (`typing`, `dataclasses`, `json`, `datetime`, `os`)

**Imported by:**
- `dependency_injection.py` — registers TokenManager as singleton
- `models/claude_model.py` — uses token manager to track costs
- `strategies/` — both strategies call token_manager methods
- `services.py` — LoggingService may log token events
- `langchain_huggingface_local.py` — logs local model usage

**Flow:**
1. Model creation includes token_manager in DI injection
2. During generation, strategy calls `token_manager.count_tokens()`
3. Strategy calls `token_manager.estimate_cost()`
4. Strategy calls `token_manager.log_usage()` to record
5. At end of session, user can call `token_manager.get_usage_summary()` to see costs

**Key insight:** This file is **optional** for basic functionality but crucial for cost control in production. All token tracking is voluntary; if you don't call the log methods, nothing gets recorded.
