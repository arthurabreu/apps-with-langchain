# Core Architecture Documentation

**📍 Documentation index** | [← Back to src/docs](../docs/index.md) | [Next: interfaces.md →](interfaces/interfaces.md)

> Complete guide to the LangChain application's core business logic, organized by dependency layers.

---

## 🗺️ Core Learning Path (10 docs, ~90 minutes)

Follow this recommended reading order to understand the entire architecture:

| # | File | Purpose | Topics |
|---|------|---------|--------|
| 1 | [interfaces.md](interfaces/interfaces.md) | Abstract contracts & data classes | Protocols, ABCs, `@dataclass`, Enums |
| 2 | [exceptions.md](exceptions/exceptions.md) | Custom exception hierarchy | Exception inheritance, error codes |
| 3 | [di/dependency_injection.md](di/dependency_injection.md) | DI container for service wiring | Singleton, factory, lambdas, `global` |
| 4 | [services.md](services/services.md) | Config, logging, user interaction | `__init__`, instance vars, logging setup |
| 5 | [utils/utils_helpers.md](utils/utils_helpers.md) | Convenience facade functions | Wrapper pattern, type hints |
| 6 | [models/model_factory.md](models/model_factory.md) | Factory pattern for model creation | Registry, type objects, `.get()` |
| 7 | [models/claude_model.md](models/claude_model.md) | Claude model implementation | ABC inheritance, strategies, `@property` |
| 8 | [strategies/standard_generation.md](strategies/standard_generation.md) | Synchronous generation | `.invoke()`, exception wrapping |
| 9 | [strategies/streaming_generation.md](strategies/streaming_generation.md) | Streaming generation | Generators, threading, fallbacks |

**Total time:** ~90 minutes for complete path

---

## 📚 Optional Reference Docs

Explore these when you need specific knowledge:

| File | When to Read | Topics |
|------|-------------|--------|
| [utils/token_utils.md](utils/token_utils.md) | Implementing token tracking | `@classmethod`, `Tuple`, JSON, `gc.collect()` |
| [hf_local/langchain_huggingface_local.md](hf_local/langchain_huggingface_local.md) | Local model support | Threading, lazy init, context managers, `.del()` |
| [models/MINIMAX_GUIDE.md](models/MINIMAX_GUIDE.md) | Using alternative open-source models | MiniMax, HuggingFace integration |

---

## 📊 Dependency Graph

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

## 🔗 File Navigation

Each file has navigation links at the top and bottom:

**Header Example:**
```
**📍 Reading Order:** #2 of 10 core docs | [← Back to Index](index.md) | [Next: exceptions.md →](exceptions.md)
```

**Footer Example:**
```
**[← Previous](interfaces.md) | [Back to Index](index.md) | [Next →](exceptions.md)**
*Read next: exceptions.md — understand the error hierarchy*
```

Click any link to jump to that file. Use "Back to Index" to return here anytime.

---

## 🎯 How Everything Fits Together

1. **User runs `python src/main.py`**
2. **main.py calls `get_container()`** → returns DI container singleton
3. **Container initializes services** via `_setup_default_services()`:
   - ConfigurationManager (reads .env)
   - LoggingService (setup logging)
   - ApiKeyValidator, TokenManager, ConsoleUserInteraction
   - ModelFactory (with all above)
4. **main.py gets ModelFactory** from container
5. **Factory creates ClaudeModel** with DI-injected dependencies
6. **ClaudeModel.generate()** selects strategy (standard or streaming)
7. **Strategy calls model.invoke()** or **model.stream()** → LangChain's ChatAnthropic
8. **Result bubbles back** → GenerationResult with tokens, cost, metadata

---

## 🧪 Testing & Mocking

The DI container is your test seam:

```python
from src.core.dependency_injection import reset_container, get_container
from src.core.interfaces import IUserInteraction

# In test:
reset_container()
container = get_container()
container.register_instance(IUserInteraction, MockUserInteraction())
# Now code using container gets your mock
```

---

## 💡 Key Python Concepts You'll Encounter

- **`__init__` and `self`** — Python's explicit constructor and instance reference
- **`@dataclass`** — Auto-generates constructors and methods
- **Protocols vs ABC** — Structural vs explicit interface contracts
- **`global` keyword** — Modify module-level variables
- **Lambdas** — Short inline functions
- **Type hints** — Optional documentation (not enforced)
- **Generators** — Functions that yield values lazily
- **Context managers** — `with` statement for resource cleanup
- **Decorators** — `@property`, `@classmethod`, `@abstractmethod`
- **f-strings** — String formatting with embedded variables

See [DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md) for detailed Python explanations.

---

## 🚀 Quick Start Navigation

**Just want to understand the basics?**
Start with: [interfaces.md](interfaces/interfaces.md) → [exceptions.md](exceptions/exceptions.md) → [di/dependency_injection.md](di/dependency_injection.md)
(~20 minutes)

**Want to understand specific features?**
- Token tracking? → [utils/token_utils.md](utils/token_utils.md)
- Model creation? → [models/model_factory.md](models/model_factory.md)
- Generation strategies? → [strategies/](strategies/)

**Want Python explanations?**
→ [DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md)

---

**[Back to src/docs →](../docs/index.md) | [Start Learning →](interfaces/interfaces.md)**

*Estimated reading time: 90 minutes for complete core learning path*
