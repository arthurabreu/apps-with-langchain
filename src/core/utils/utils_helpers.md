# utils.py

**📍 Reading Order:** #6 of 10 core docs | [← Back to Index](index.md) | [Next: models/model_factory.md →](models/model_factory.md)

> Convenience functions that delegate to the DI container. Provides backward compatibility and a simpler interface.

## What This File Does

This is a **facade** over the DI container. Instead of writing:

```python
from src.core.dependency_injection import get_container
container = get_container()
user_interaction = container.get_user_interaction()
should_continue = user_interaction.prompt_continue()
```

You can just write:

```python
from src.core.utils import prompt_continue
should_continue = prompt_continue()
```

It's a thin convenience layer for common operations. Used mainly by `main.py` and examples.

---

## Functions

| Name | What It Does |
|------|-------------|
| `prompt_continue() -> bool` | Ask user to continue or skip (delegates to container) |
| `create_claude_model(model_name, temperature, max_tokens)` | Create a Claude model with defaults (delegates to factory) |

---

## prompt_continue()

```python
def prompt_continue() -> bool:
    """
    Prompt user to continue to next generation or skip.

    Returns:
        bool: True to continue, False to skip
    """
    container = get_container()
    user_interaction = container.get_user_interaction()
    return user_interaction.prompt_continue()
```

**Usage:**
```python
if prompt_continue():
    print("Continuing...")
else:
    print("Skipping...")
```

**Behind the scenes:**
1. Get the DI container
2. Ask it for ConsoleUserInteraction
3. Call `prompt_continue()` on that instance
4. Return the result

This is just a wrapper so callers don't need to know about the DI container.

---

## create_claude_model()

```python
def create_claude_model(
    model_name: str = "claude-3-haiku-20240307",
    temperature: float = 0.2,
    max_tokens: int = 512
):
    """
    Create a Claude model with default configuration.
    Backward compatibility function.

    Args:
        model_name: Claude model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Configured Claude model instance
    """
    container = get_container()
    factory = container.get_model_factory()

    config = ModelConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return factory.create_model("anthropic", config)
```

**Usage:**
```python
model = create_claude_model(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.5,
    max_tokens=1024
)
result = model.generate("Hello, world!")
print(result.content)
```

**Behind the scenes:**
1. Get the DI container
2. Ask it for ModelFactory
3. Create a ModelConfig with provided arguments
4. Call factory.create_model("anthropic", config)
5. Return the created model

This is a convenience for quick experimentation without dealing with config objects directly.

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `from .dependency_injection import get_container` | `import dependency_injection.getContainer` | Imports |
| `def prompt_continue() -> bool:` | `fun promptContinue(): Boolean` | Function definition |
| `def create_claude_model(...):` | `fun createClaudeModel(...)` | Function definition |
| `container = get_container()` | `val container = getContainer()` | Getting singleton |
| `container.get_user_interaction()` | `container.getUserInteraction()` | Method call |
| Default parameters | `fun create(..., temperature: Float = 0.2f)` | Default argument values |
| Return type annotation `-> bool` | `: Boolean` | Return type |
| Docstring (triple quote) | KDoc `/** ... */` | Documentation |

---

## How It Connects to Other Files

**Imports from:**
- `dependency_injection.py` — `get_container()`
- `interfaces.py` — `ModelConfig`

**Imported by:**
- `main.py` — calls `prompt_continue()` and `create_claude_model()`
- Examples in documentation or demos
- Tests that need quick model creation

**Flow:**
1. User calls `prompt_continue()` or `create_claude_model()`
2. Function gets container
3. Function asks container for service
4. Container creates or returns cached service
5. Function calls method on service
6. Result bubbles back to user

**When to use:**
- Quick experimentation in scripts
- Backward compatibility with older code
- Simple operations that don't need full DI container access

**When NOT to use:**
- In production code that needs testability (get container yourself and mock services)
- Complex flows (get container once, reuse services)
- When you need to inject mocks (access container directly)

---

## Why This File Exists

In the old codebase, models and containers were simpler. This file provides a migration path for code that was written before DI was introduced. New code should access the container directly:

```python
# Old style (convenience functions)
model = create_claude_model()

# New style (explicit, more testable)
container = get_container()
factory = container.get_model_factory()
config = ModelConfig(model_name="claude-3-haiku-20240307")
model = factory.create_model("anthropic", config)
```

Both work; the second is more explicit and easier to test.

---

**[← Previous](services.md) | [Back to Index](index.md) | [Next →](models/model_factory.md)**

*Read next: models/model_factory.md — understand how models are created*
