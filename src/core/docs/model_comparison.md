# model_comparison.py

> Utilities for comparing different language models side-by-side on the same prompts. Measures performance, latency, and quality.

## What This File Does

This module lets you **benchmark models**: run the same prompt on multiple models, measure response time and quality, and summarize results. Useful for:

- Deciding which model fits your use case
- Measuring latency differences (Claude vs. local HF)
- Comparing response quality
- Cost vs. quality tradeoffs

Currently supports optional models (imports try/except for optional dependencies):
- Local HuggingFace models (if transformers available)
- Claude (if anthropic SDK available)

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `ModelComparison` | class | Main comparison orchestrator |
| `.run_comparison(prompts)` | method | Test models with given prompts, record results |
| `.print_summary()` | method | Display aggregated statistics |

---

## ModelComparison Class

```python
class ModelComparison:
    def __init__(self):
        self.results = []  # List of comparison results
```

### run_comparison()

```python
def run_comparison(self, prompts: List[Dict[str, str]] = None):
    """
    Run comparison with different models.

    Args:
        prompts: List of dicts with 'name' and 'prompt' keys
                 If None, uses default Kotlin-related prompts

    Returns:
        None (stores results in self.results)
    """
    if prompts is None:
        prompts = [
            {
                "name": "Kotlin Coroutine",
                "prompt": "Write a Kotlin function that uses coroutines..."
            },
            {
                "name": "Kotlin Palindrome",
                "prompt": "Write a Kotlin function that checks if a string is a palindrome..."
            }
        ]
```

**Flow:**
1. Detect available models via **dynamic imports** with try/except
2. For each model:
   - For each prompt:
     - Measure time with `datetime`
     - Call `model.generate(prompt)`
     - Record result (name, prompt, duration, response length, timestamp)
3. Print results and summary

### Dynamic Imports (Try/Except Pattern)

```python
models_to_test = []

try:
    from .langchain_huggingface_local import LocalHuggingFaceModel
    models_to_test.append(("Local Hugging Face", LocalHuggingFaceModel()))
except Exception as e:
    print(f"[SKIP] Local model not available: {e}")

try:
    from .claude_model import ClaudeModel
    models_to_test.append(("Anthropic Claude", ClaudeModel()))
except Exception as e:
    print(f"[SKIP] Claude model not available: {e}")
```

**Pattern:** If a dependency isn't available, catch the import error and skip it. This makes optional dependencies truly optional—app doesn't crash if transformers isn't installed.

**Kotlin equivalent:**
```kotlin
val models = mutableListOf<Pair<String, Model>>()
try {
    val localModel = LocalHuggingFaceModel()
    models.add("Local Hugging Face" to localModel)
} catch (e: Exception) {
    println("[SKIP] Local model not available: $e")
}
```

### Data Collection

For each model + prompt combo:
```python
start_time = datetime.now()
response = model.generate(prompt_info['prompt'])
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

self.results.append({
    "model": model_name,
    "prompt": prompt_info['name'],
    "duration": duration,
    "response_length": len(response),
    "timestamp": datetime.now().isoformat()
})
```

**Stores:**
- Which model
- Which prompt
- How long it took (seconds)
- Response length (chars)
- When it ran (ISO timestamp)

### print_summary()

```python
def print_summary(self):
    """Print aggregated statistics by model."""
    from collections import defaultdict
    model_stats = defaultdict(list)  # Auto-create empty list if key missing

    for result in self.results:
        model_stats[result['model']].append(result)

    for model, results in model_stats.items():
        avg_time = sum(r['duration'] for r in results) / len(results)
        avg_length = sum(r['response_length'] for r in results) / len(results)

        print(f"\n{model}:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average response length: {avg_length:.0f} chars")
        print(f"  Tests completed: {len(results)}")
```

**Uses `defaultdict`:** If key doesn't exist, auto-creates empty list. No need for `if key not in dict`.

**Kotlin equivalent:**
```kotlin
val modelStats = mutableMapOf<String, MutableList<Result>>()
for (result in results) {
    modelStats.getOrPut(result.model) { mutableListOf() }.add(result)
}
```

---

## Example Usage

```python
comparison = ModelComparison()

# Run with custom prompts
custom_prompts = [
    {"name": "Hello", "prompt": "Say hello in 5 words"},
    {"name": "Code", "prompt": "Write a function to reverse a string"},
]
comparison.run_comparison(custom_prompts)

# Print results
comparison.print_summary()
```

**Output:**
```
============================================================
Comparison Summary
============================================================

Anthropic Claude:
  Average time: 2.45s
  Average response length: 285 chars
  Tests completed: 2

Local Hugging Face:
  Average time: 15.32s
  Average response length: 192 chars
  Tests completed: 2
```

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `List[Dict[str, str]]` | `List<Map<String, String>>` | prompts parameter |
| `datetime.now()` | `Instant.now()` or `LocalDateTime.now()` | timing |
| `(end - start).total_seconds()` | `(end - start).seconds()` | duration calculation |
| `from collections import defaultdict` | `mutableMapOf<K, MutableList<V>>()` with `.getOrPut()` | model_stats |
| `defaultdict(list)` | No direct equiv; use `.getOrPut(key) { mutableListOf() }` | Auto-create empty list |
| Try/except for import | Try/catch or reflection | Dynamic imports |
| `f"{model}: {duration:.2f}s"` | `"$model: ${String.format("%.2f", duration)}s"` | String formatting |
| `.append(dict)` | `.add(dict)` | results list |
| `enumerate()` | `.withIndex()` or `.forEachIndexed()` | Loop with index |

---

## How It Connects to Other Files

**Imports from:**
- Standard library (`typing`, `datetime`, `collections`)
- Conditional/dynamic: `langchain_huggingface_local`, `models/claude_model`

**Imported by:**
- Rarely; usually run standalone from main.py or tests

**Flow:**
1. Main code (or test) creates `ModelComparison()`
2. Calls `run_comparison(prompts)`
3. Module dynamically imports available models (Claude, Local HF)
4. For each model × prompt combo:
   - Measures time, captures response
   - Stores results
5. Calls `print_summary()` to display stats

**Known issue:** `ClaudeModel` import path may be stale if code structure changes. If import fails, check actual path and update line 50 in model_comparison.py.

---

## Key Patterns

**Dynamic imports (optional dependencies):**
```python
try:
    from .some_module import SomeClass
    instances.append(SomeClass())
except Exception as e:
    print(f"[SKIP] {e}")
```

**defaultdict for auto-grouping:**
```python
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    groups[item.category].append(item)
# No need to check if category exists
```

**Elapsed time measurement:**
```python
start = datetime.now()
# ... do work ...
elapsed = (datetime.now() - start).total_seconds()
```

**Result accumulation:**
```python
results = []
for ... :
    results.append({
        "name": ...,
        "value": ...
    })
# Later, iterate and aggregate
```
