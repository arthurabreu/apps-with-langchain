# Developer Guide: Python for Android/Kotlin Devs

**📍 Learning guide** | [← Back to Index](index.md) | [Next: Core Architecture →](index.md#recommended-reading-order)

> As a senior Android/Kotlin developer, you already understand software architecture, dependency injection, and design patterns. This guide explains Python-specific idioms and why the project is structured the way it is.

---

## 1. Entry Points and Module Execution

### Python: `if __name__ == "__main__":`

```python
# src/main.py
def main():
    container = get_container()
    # ... start app ...

if __name__ == "__main__":
    main()
```

**What it does:** This code runs only when you execute `python src/main.py` directly, NOT when you import this file as a module. It's Python's way of saying "this is the entry point."

**Why this pattern?**
- If another file imports from `main.py`, the `if __name__ == "__main__":` block doesn't run
- Prevents side effects when the module is imported
- Lets you test functions without executing the main flow

**Kotlin equivalent:** You don't need this in Kotlin/Android—the `MainActivity` is marked as the entry point in `AndroidManifest.xml`.

---

## 2. Classes and Constructors

### Python: `__init__` and `self`

```python
class ConfigurationManager:
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level  # Instance variable
        self._config = {}           # Private by convention (underscore)
```

**Key differences from Kotlin:**

| Feature | Python | Kotlin |
|---------|--------|--------|
| Constructor | `def __init__(self, ...):` | `constructor(...)` or `init {}` |
| Instance vars | `self.var = value` (set in `__init__`) | `val/var name: Type` (declared in constructor) |
| `self` param | Explicitly first parameter in every method | Implicit (you never write `this` unless shadowed) |
| Private vars | `_name` (convention) or `__name__` (name mangling) | `private` keyword |

**Why `self` is explicit in Python:**
```python
class MyClass:
    def method(self):
        # 'self' refers to this instance
        print(self.x)  # Access instance variable
    
obj = MyClass()
obj.method()  # Python passes obj as 'self' automatically
# Equivalent to: MyClass.method(obj)
```

---

## 3. Data Classes

### Python: `@dataclass`

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.2        # Default value
    max_tokens: int = 512
```

**Automatically generates:**
- `__init__(self, model_name, temperature=0.2, max_tokens=512)`
- `__repr__()` for nice printing
- `__eq__()` for equality comparison

**Usage:**
```python
config = ModelConfig(model_name="claude-3-haiku")
print(config)  # ModelConfig(model_name='claude-3-haiku', temperature=0.2, max_tokens=512)
config2 = ModelConfig(model_name="claude-3-haiku")
assert config == config2  # True! __eq__ works
```

**Kotlin equivalent:** `data class ModelConfig(...)`

---

## 4. Interfaces: ABC vs Protocol

Python provides two ways to define contracts:

### ABC (Abstract Base Class) — Explicit Inheritance Required

```python
from abc import ABC, abstractmethod

class ILanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class ClaudeModel(ILanguageModel):
    def generate(self, prompt: str) -> str:
        # Must implement, or get error at instantiation
        return "..."
```

**When to use:** When you want to enforce a contract and subclasses must explicitly declare inheritance.

**Kotlin equivalent:** `abstract class`

### Protocol — Structural Typing (No Explicit Inheritance)

```python
from typing import Protocol

class ITokenCounter(Protocol):
    def count_tokens(self, text: str) -> int:
        ...

class MyCounter:
    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # No explicit inheritance!

counter: ITokenCounter = MyCounter()  # Works! Type checker sees the method
```

**When to use:** For loose coupling, multiple composition, or when you don't control the class being used.

**Key difference from Kotlin:** Kotlin interfaces require explicit `implements`, but Protocols don't. The type checker just verifies the methods exist.

---

## 5. Dependency Injection (DI)

This project uses a **manual DI Container** (not a framework). Here's how it works:

### Registration (Setup Phase)

```python
# src/core/dependency_injection.py
class DIContainer:
    def __init__(self):
        self._services = {}      # Registry of services
        self._singletons = {}    # Cache of created singletons
    
    def register_singleton(self, interface, impl):
        """Create once, reuse forever"""
        self._services[interface.__name__] = ('singleton', impl)
    
    def register_factory(self, interface, factory_func):
        """Create new instance every time"""
        self._services[interface.__name__] = ('factory', factory_func)

# In _setup_default_services():
self.register_singleton(ConfigurationManager, ConfigurationManager)
self.register_factory(
    LoggingService,
    lambda: LoggingService(self.get(ConfigurationManager))
)
```

### Retrieval (Use Phase)

```python
# In main.py
container = get_container()
factory = container.get_model_factory()
model = factory.create_model("anthropic", config)
```

**Why manual DI instead of a framework?**
- **Simplicity:** No external dependencies, easier to understand
- **Testability:** Reset container in tests, inject mocks easily
- **Control:** You see exactly how services are wired

**Kotlin equivalent:**
- **Hilt:** `@HiltViewModel`, `@Inject`
- **Koin:** `val myModule = module { single { ConfigurationManager() } }`

Our manual approach is more explicit (longer code) but more transparent.

---

## 6. Type Hints (Not Enforced at Runtime)

### Python's Type System is Optional

```python
def calculate(x: int, y: int) -> int:
    """Type hints say: expects ints, returns int"""
    return x + y

result = calculate("hello", "world")  # No error! Python doesn't check at runtime
# Type checkers (mypy, Pylint) would warn, but the code runs
```

**Key difference:** Kotlin/Java type checking happens at **compile time** (before running). Python's type hints are checked by external tools, not the runtime.

**Best practice:** Always add type hints, then run `mypy` to catch type errors:
```bash
mypy src/  # Check all Python files for type errors
```

---

## 7. Async and Awaitable (Coroutines)

### Python: `async def` / `await`

```python
async def fetch_data(api_url: str) -> str:
    response = await http_client.get(api_url)  # Non-blocking wait
    return response.body

# Call async function
result = asyncio.run(fetch_data("..."))  # Must use asyncio.run() or await from async context
```

**Kotlin equivalent:** `suspend fun`

**Key difference:** In Kotlin/Android, you use `launch {}` or `async {}` builders. In Python, you use `asyncio.run()` or `await` from another async function.

**Current project status:** Most code is **synchronous** (no `async`/`await`) for simplicity in a CLI. The architecture is ready to add async if needed.

---

## 8. Project Structure and Orchestration

```
src/
├── main.py                      # Entry point (if __name__ == "__main__")
├── core/
│   ├── dependency_injection.py  # DI Container (like Hilt)
│   ├── interfaces.py            # Contracts (ABC & Protocol)
│   ├── services.py              # ConfigManager, Logger, Validators
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── models/
│   │   ├── claude_model.py      # Claude implementation
│   │   └── model_factory.py     # Factory (Open/Closed Principle)
│   └── strategies/
│       ├── standard_generation.py
│       └── streaming_generation.py
└── docs/                        # This file + other docs
```

**Architectural pattern:** "Thin Entry Point" + Dependency Injection

1. **`main.py`** calls `get_container()` (very simple, no business logic)
2. **`DIContainer`** wires all services (like Hilt's `@Inject`)
3. **`ModelFactory`** creates models (Strategy Pattern + Factory Pattern)
4. **Strategies** handle generation (Pluggable behavior)
5. **Services** provide utilities (ConfigManager, Logger, TokenManager)

**Kotlin equivalent:**
- `main()` ← `main.py`
- Hilt/Koin ← `DIContainer`
- Factory pattern ← `ModelFactory`
- Strategy pattern ← Generation strategies

---

## 9. Key Python Idioms

### List Comprehensions

```python
# Old way
results = []
for item in items:
    if item > 0:
        results.append(item * 2)

# Pythonic way
results = [item * 2 for item in items if item > 0]
```

### Dictionary Defaults with `.get()`

```python
config = {}
value = config.get("key", "default")  # Returns "default" if key doesn't exist
# vs config["key"]  # Raises KeyError if missing
```

### F-Strings for Formatting

```python
name = "Claude"
version = 3.5
message = f"Hello, I'm {name} {version}"  # Much cleaner than format() or +
```

### Lambda Functions

```python
# Short function in one line
times_two = lambda x: x * 2
result = times_two(5)  # 10

# Common in DI registration
self.register_factory(Logger, lambda: Logger(get_container().get(ConfigManager)))
```

---

## 10. Debugging Tips

### Print vs Logger

```python
# Avoid this in production
print("Debug info")  # Can't disable, no timestamp, no level

# Use this
logger.debug("Debug info")   # Can control log level
logger.info("Info message")
logger.warning("Warning")
logger.error("Error")
```

### Type Checking

```bash
# Run mypy to catch type errors before runtime
pip install mypy
mypy src/
```

### Interactive Debugging

```python
import pdb; pdb.set_trace()  # Breakpoint (Python 3.7+)
# or
breakpoint()  # Python 3.7+ shorthand
```

---

## Summary: Python vs Kotlin/Android

| Task | Python | Kotlin |
|------|--------|--------|
| Define class | `class MyClass:` | `class MyClass` |
| Constructor | `def __init__(self, ...):` | `constructor(...)` or `init {}` |
| Instance variable | `self.var = value` | `val/var name: Type` |
| Data class | `@dataclass` | `data class` |
| Abstract contract | `ABC` | `abstract class` |
| Interface/Protocol | `Protocol` | `interface` |
| Get value or default | `dict.get(key, default)` | `map[key] ?: default` |
| Dependency injection | Manual `DIContainer` | Hilt, Dagger, Koin |
| Type hints | Optional (not enforced) | Required (enforced) |
| Async function | `async def` / `await` | `suspend fun` / `await` |

---

**[← Previous](QUICK_START.md) | [Back to Index](index.md) | [Next →](index.md)**

*Next step: View the Documentation Index to start reading core architecture docs*
