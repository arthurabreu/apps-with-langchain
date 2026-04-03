# src/core/ Documentation Index

## Architecture Overview

This directory contains the **core business logic** of the LangChain application, organized in layers:

1. **Interfaces** (`interfaces.py`) — Define contracts (abstract base classes & Protocols)
2. **Exceptions** (`exceptions.py`) — Custom exception hierarchy
3. **Services** (`services.py`) — Concrete implementations (config, logging, user interaction, API validation)
4. **Dependency Injection** (`dependency_injection.py`) — Singleton/factory container wiring everything together
5. **Models** (`models/`) — Language model implementations (Claude, etc.)
6. **Strategies** (`strategies/`) — Pluggable generation approaches (standard vs. streaming)
7. **Utilities** (`utils.py`, `token_utils.py`, etc.) — Helpers for tokens, device detection, model comparison

**Design principle**: Interfaces define what services *must* do; concrete implementations in services/models provide *how* they do it. The DI container orchestrates creation and wiring.

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   (orchestration only)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   get_container() → DI       │
        │   Container instance        │
        └──────┬──────────────────┬───┘
               │                  │
               ▼                  ▼
        ┌────────────────┐  ┌──────────────────┐
        │  ModelFactory  │  │ ConfigMgr, etc.  │
        └────────┬───────┘  └──────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
    ┌─────────┐    ┌───────────┐
    │ Claude  │    │ Strategies│
    │ Model   │    └─┬──────┬──┘
    └──┬──────┘      │      │
       │      ┌──────┴┐  ┌─┴──────────┐
       │      ▼       ▼  ▼            ▼
       │   Standard  Streaming  Token  User
       │ Generation Generation Manager Interaction
       │
       └─────────────────────────────────┘
                 (all use)
```

---

## File Documentation Index

| File | Purpose | For Android Devs |
|------|---------|------------------|
| [interfaces.md](interfaces.md) | Abstract contracts & data classes | Protocol ≈ interface, ABC ≈ abstract class |
| [exceptions.md](exceptions.md) | Custom exception hierarchy | Exception ≈ Exception, inheritance tree |
| [services.md](services.md) | Config, logging, user I/O, validation | Singleton pattern, dependency injection |
| [dependency_injection.md](dependency_injection.md) | Manual DI container | No Hilt/Dagger, manual dictionary-based wiring |
| [utils.md](utils.md) | Convenience functions | Facade over DI container |
| [token_utils.md](token_utils.md) | Token counting, cost tracking | @classmethod ≈ companion object |
| [langchain_huggingface_local.md](langchain_huggingface_local.md) | Local HF model inference | Threading, context managers, lazy init |
| [model_comparison.md](model_comparison.md) | Compare models side-by-side | defaultdict, dynamic imports |
| [models/claude_model.md](models/claude_model.md) | Claude model implementation | ABC inheritance, strategy pattern dispatch |
| [models/model_factory.md](models/model_factory.md) | Factory for creating models | Registry pattern, Open/Closed principle |
| [strategies/standard_generation.md](strategies/standard_generation.md) | Sync text generation | LangChain chains, `invoke()` method |
| [strategies/streaming_generation.md](strategies/streaming_generation.md) | Streaming text generation | Generators, real-time output |

---

## Python → Kotlin Quick Reference

| Pattern | Python | Kotlin | Where Used |
|---------|--------|--------|-----------|
| Abstract base class | `from abc import ABC`; `class MyModel(ABC):` | `abstract class MyModel` | `ILanguageModel` in interfaces.py |
| Protocol (structural) | `from typing import Protocol`; `class MyProto(Protocol):` | `interface MyInterface` (no explicit impl needed if struct matches) | Token managers, user interaction in interfaces.py |
| Data class | `@dataclass` decorator on `class ModelConfig:` | `data class ModelConfig(...)` | `ModelConfig`, `GenerationResult` |
| Enum | `from enum import Enum`; `class GenerationStrategy(Enum):` | `enum class GenerationStrategy` | Strategy selection |
| Optional type | `Optional[str]` = `str \| None` | `String?` | Config fields |
| Class method | `@classmethod def method(cls):` | `companion object { fun method() }` | `TokenManager.resolve_model()` |
| Module-level singleton | `_container: Optional[DIContainer] = None` | Kotlin `object` or `companion object` | DI container in dependency_injection.py |
| Lambda / function ref | `lambda: MyClass()` | `{ MyClass() }` lambda or `::MyClass` reference | Factory functions in DI |
| TypeVar + Type[T] | `T = TypeVar('T')` then `func(interface: Type[T])` | `inline fun <reified T>` | Generic registration in DIContainer |
| Generator / Iterator | `for item in generator:` from function with `yield` | `Sequence<T>` or `Iterator<T>` | Streaming strategy chunks |
| Default param | `def func(logger: Logger = None):` | `fun func(logger: Logger? = null)` | Service constructors |
| `vars(obj)` for serialization | `vars(usage)` → dict of all fields | `Gson().toJson(obj)` or `dataclass.copy()` | TokenUsage in token_utils.py |
| `**kwargs` unpacking | `MyClass(**dict_args)` | `MyClass(dict_args.mapToParams())` or named params | Model creation with config dicts |
| `if __name__ == "__main__"` | Module-level guard for running standalone | `fun main(args: Array<String>)` | Token_utils.py example at bottom |
| Context manager | `with SomeClass() as resource:` / `__enter__`, `__exit__` | `Closeable.use { resource -> }` | LocalHuggingFaceModel cleanup |
| `@property` | `@property def name(self):` — read-only computed property | Kotlin `val name: String` (computed body) | `ClaudeModel.provider` |

---

## How Everything Fits Together

1. **User runs `python src/main.py`** → calls orchestration in main.py
2. **main.py calls `get_container()`** → returns singleton DI container
3. **Container is initialized once** with `_setup_default_services()`:
   - Creates `ConfigurationManager` (reads .env)
   - Creates `LoggingService` (setup logging)
   - Creates `ApiKeyValidator`, `TokenManager`, `ConsoleUserInteraction`
   - Creates `ModelFactory` with all the above
4. **main.py asks container for `ModelFactory`** → gets fully wired instance
5. **main.py calls `factory.create_model("anthropic", config)`** → factory validates API key, creates `ClaudeModel` with DI-injected token manager, logger, etc.
6. **ClaudeModel.generate(prompt)** → picks strategy (standard or streaming), delegates to it
7. **Strategy calls model.invoke() or model.stream()** → LangChain's ChatAnthropic does the real work
8. **Result bubbles back through layers** → `GenerationResult` with tokens, cost, metadata

---

## Testing & Mocking

The DI container is your test seam:

```python
from src.core.dependency_injection import reset_container, get_container
from src.core.interfaces import IUserInteraction

# In your test:
reset_container()  # Clear singleton
container = get_container()
container.register_instance(IUserInteraction, MockUserInteraction())
# Now any code that calls container.get_user_interaction() gets your mock
```

---

## Recommended Reading Order

**Core Learning Path** (start here):

| # | File | Purpose | Read time |
|---|------|---------|-----------|
| 1 | [index.md](index.md) | Start here — architecture overview | 5 min |
| 2 | [interfaces.md](interfaces.md) | Understand all contracts & data classes | 10 min |
| 3 | [exceptions.md](exceptions.md) | Learn the error hierarchy | 8 min |
| 4 | [dependency_injection.md](dependency_injection.md) | See how DI container wires everything | 12 min |
| 5 | [services.md](services.md) | Understand concrete implementations | 10 min |
| 6 | [utils.md](utils.md) | Learn convenience functions | 5 min |
| 7 | [models/model_factory.md](models/model_factory.md) | Understand how models are created | 8 min |
| 8 | [models/claude_model.md](models/claude_model.md) | See a specific model implementation | 10 min |
| 9 | [strategies/standard_generation.md](strategies/standard_generation.md) | Learn synchronous generation | 10 min |
| 10 | [strategies/streaming_generation.md](strategies/streaming_generation.md) | Learn streaming generation | 10 min |

**Optional/Reference** (explore as needed):

| File | When to read |
|------|-------------|
| [token_utils.md](token_utils.md) | When you need to understand token/cost tracking |
| [langchain_huggingface_local.md](langchain_huggingface_local.md) | When exploring local model support (educational) |
| [model_comparison.md](model_comparison.md) | When benchmarking different models |

**Estimated total reading time:** 90 minutes for core path

---

## Navigation Tips

Each file has **navigation links** at the top and bottom:
- Click "Next →" to jump to the recommended next file
- Click "← Previous" to go back
- Click "📍 Back to Index" to return here anytime
- Look for `[Read next: filename.md →]` at bottom of each file

**Pro tip:** Open this index in one browser tab, then open individual files in other tabs as you read. Use the navigation links to move through the learning path.
