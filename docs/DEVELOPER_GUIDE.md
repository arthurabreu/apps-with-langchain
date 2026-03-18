# Developer Guide: Python for Android Devs

Welcome! If you are a senior Android/Kotlin developer, this guide will help you understand the Python patterns used in this project by mapping them to concepts you already know.

## 1. Structural Comparison

| Feature | Python | Kotlin / Android |
|---------|--------|-----------------|
| **Main Class** | `main.py` + `if __name__ == "__main__":` | `MainActivity` or `Application.onCreate()` |
| **Data Class** | `@dataclass class User:` | `data class User(...)` |
| **Interfaces** | `abc.ABC` or `typing.Protocol` | `interface` or `abstract class` |
| **Constructor** | `def __init__(self, ...):` | `constructor(...)` or `init { ... }` |
| **Null Safety** | `Optional[str]` or `str \| None` | `String?` |
| **DI Container** | `DIContainer` (in this project) | Dagger, Hilt, or Koin |
| **Singleton** | Module-level variables or DI managed | `object` or `@Singleton` |
| **Async** | `async def` / `await` | `suspend` / `await` |

## 2. Dependency Injection (DI)

In this project, we use a manual DI Container found in `src/core/dependency_injection.py`.

- **Kotlin (Koin):**
  ```kotlin
  val myModule = module {
      single { ConfigurationManager() }
      factory { LoggingService(get()) }
  }
  ```
- **Python (Our DI):**
  ```python
  self.register_singleton(ConfigurationManager, ConfigurationManager)
  self.register_factory(LoggingService, lambda: LoggingService(self.get(ConfigurationManager)))
  ```

## 3. Interfaces and Protocols

Python doesn't have a strict `interface` keyword, so we use two things:
1. **ABC (Abstract Base Class):** Forces inheritance. Like an `abstract class`.
2. **Protocol:** Structural typing. If a class has the method, it "implements" it. Like a Kotlin interface but more flexible.

## 4. Pydantic & Dataclasses vs Kotlin Data Classes

We use `@dataclass` for simple data holders like `ModelConfig`. 
- They automatically generate `__init__`, `__repr__`, and `__eq__`.
- Exactly like Kotlin `data class`.

## 5. Async and Coroutines

Python's `asyncio` is very similar to Kotlin Coroutines.
- `async def` = `suspend fun`
- `await` = `await()` or simply calling a suspend function.
- In this project, most calls are synchronous for simplicity in the CLI, but the architecture is ready for async.

## 6. Project Orchestration

This project follows a "Thin Entry Point" pattern:
1. `main.py`: The Launcher.
2. `src/core/cli_service.py`: The Orchestrator (MainActivity logic).
3. `src/core/models/`: The Domain/Data layer (Model implementations).
4. `src/core/dependency_injection.py`: The Glue (Hilt/Koin).

---
*Senior Note: Python is dynamically typed. We use "Type Hints" (like `: str`) to help the IDE, but they aren't enforced at runtime like in Kotlin.*
