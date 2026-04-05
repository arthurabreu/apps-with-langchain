# dependency_injection.md

**📍 Reading Order:** #4 of 10 core docs | [← Back to Index](index.md) | [Next: services.md →](services.md)

> Manual dependency injection container for service creation and wiring. No Hilt/Dagger—just a dictionary-based registry.

## What This File Does

Dependency Injection (DI) is a pattern where you pass dependencies to objects instead of having them create their own. This file provides:

1. **DIContainer** — a manual DI registry (dictionary-based)
2. **Global singleton instance** — accessed via `get_container()`
3. **Service registration methods** — `register_singleton()`, `register_factory()`, `register_instance()`
4. **Service retrieval** — `get()` with automatic instantiation

Think of the container as a **service registry** that knows how to construct and wire up all services. In Android, this is like Hilt/Dagger, but manual and minimal.

**Key principle:** Set up all wiring in one place (`_setup_default_services()`) so `main.py` just asks for what it needs without knowing how it's constructed.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `DIContainer` | class | Manual DI registry; creates and wires services |
| `.__init__()` | method | Initialize container, call `_setup_default_services()` |
| `._setup_default_services()` | method | Register all default services |
| `.register_singleton(interface, impl)` | method | Register a class as singleton (create once) |
| `.register_factory(interface, func)` | method | Register a factory function (call every time) |
| `.register_instance(interface, instance)` | method | Register a pre-made instance (for testing) |
| `.get(interface)` | method | Get a service instance |
| `.get_token_manager()` | method | Convenience: get TokenManager |
| `.get_user_interaction()` | method | Convenience: get ConsoleUserInteraction |
| `.get_model_factory()` | method | Convenience: get ModelFactory |
| `.get_config_manager()` | method | Convenience: get ConfigurationManager |
| `.get_logging_service()` | method | Convenience: get LoggingService |
| `get_container()` | function | Get global container (singleton pattern) |
| `reset_container()` | function | Reset global container (for testing) |

---

## DIContainer (Detailed)

### Initialization

```python
class DIContainer:
    def __init__(self):
        self._services: Dict[str, Any] = {}       # Registry of service info
        self._singletons: Dict[str, Any] = {}      # Cache of created singletons
        self._setup_default_services()             # Wire everything up

    def _setup_default_services(self) -> None:
        # Register all default services in order of dependencies
        # 1. Configuration (no dependencies)
        # 2. Logging (depends on Configuration)
        # 3. Services (ValidationCan depend on nothing or logging)
        # 4. ModelFactory (depends on all above)
```

### Service Registration

**Singleton (create once, reuse):**
```python
def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
    self._services[interface.__name__] = ('singleton', implementation)
    # Example: register_singleton(ConfigurationManager, ConfigurationManager)
    # First call to .get() creates it; subsequent calls return same instance
```

**Factory (create every time):**
```python
def register_factory(self, interface: Type[T], factory_func) -> None:
    self._services[interface.__name__] = ('factory', factory_func)
    # Example: register_factory(ModelFactory, lambda: ModelFactory(...))
    # Every call to .get() calls factory_func()
```

**Instance (pre-made, for testing):**
```python
def register_instance(self, interface: Type[T], instance: T) -> None:
    self._singletons[interface.__name__] = instance
    # Example: register_instance(IUserInteraction, MockUserInteraction())
    # Immediate use; useful in tests to inject mocks
```

### Service Retrieval

```python
def get(self, interface: Type[T]) -> T:
    service_name = interface.__name__

    # 1. Check if already instantiated (singleton)
    if service_name in self._singletons:
        return self._singletons[service_name]

    # 2. Check if registered
    if service_name not in self._services:
        raise ValueError(f"Service {service_name} not registered")

    service_type, service_impl = self._services[service_name]

    # 3. Create or call factory
    if service_type == 'singleton':
        instance = service_impl()  # Instantiate class
        self._singletons[service_name] = instance  # Cache it
        return instance
    elif service_type == 'factory':
        return service_impl()  # Call factory function every time
```

---

## Default Service Wiring (_setup_default_services)

Here's what gets registered on startup:

```python
def _setup_default_services(self) -> None:
    # 1. Configuration (foundation)
    self.register_singleton(ConfigurationManager, ConfigurationManager)

    # 2. Logging (depends on Configuration)
    self.register_factory(
        LoggingService,
        lambda: LoggingService(self.get(ConfigurationManager))
    )

    # 3. Base services
    self.register_singleton(ApiKeyValidator, ApiKeyValidator)
    self.register_singleton(TokenManager, TokenManager)

    # 4. User interaction (depends on Logging)
    self.register_factory(
        ConsoleUserInteraction,
        lambda: ConsoleUserInteraction(
            self.get(LoggingService).get_logger("user_interaction")
        )
    )

    # 5. ModelFactory (depends on everything)
    self.register_factory(
        ModelFactory,
        lambda: ModelFactory(
            config_manager=self.get(ConfigurationManager),
            api_key_validator=self.get(ApiKeyValidator),
            token_manager=self.get(TokenManager),
            user_interaction=self.get(ConsoleUserInteraction),
            logging_service=self.get(LoggingService)
        )
    )
```

**Dependency order matters!** ConfigurationManager has no deps, so register it first. LoggingService depends on ConfigurationManager, so it comes after. ModelFactory depends on all, so it goes last.

---

## Global Container (Singleton Pattern)

```python
_container: Optional[DIContainer] = None  # Module-level global

def get_container() -> DIContainer:
    """Get or create the global container."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container

def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _container
    _container = None
```

**Usage:**
```python
# In main.py
container = get_container()
factory = container.get_model_factory()

# In tests
reset_container()  # Clear old container
container = get_container()  # Get fresh one
container.register_instance(IUserInteraction, MockInteraction())
# Now any code calling get_container() gets the test version
```

---

## Convenience Getter Methods

The container provides shortcuts:

```python
def get_token_manager(self) -> ITokenManager:
    return self.get(TokenManager)

def get_user_interaction(self) -> IUserInteraction:
    return self.get(ConsoleUserInteraction)

def get_model_factory(self) -> ModelFactory:
    return self.get(ModelFactory)

# ... and so on for other services
```

These save typing and hide implementation details from callers.

---

## Key Python Concepts (This File)

### 1. **TypeVar — Generic Type Parameters**

```python
from typing import TypeVar

T = TypeVar('T')  # A placeholder for any type

def get(self, interface: Type[T]) -> T:
    # Means: "whatever type you pass in, I return the same type"
    # get(ConfigurationManager) -> ConfigurationManager instance
    # get(TokenManager) -> TokenManager instance
```

`TypeVar('T')` creates a **type variable** that represents "some type." When you see `Type[T] -> T`, it means "pass a class, get an instance of that class." Helps the type checker track types through generic functions.

### 2. **`global` Keyword — Module-Level Variables**

```python
_container: Optional[DIContainer] = None  # Module-level variable

def get_container() -> DIContainer:
    global _container  # Tell Python this function modifies the global variable
    if _container is None:
        _container = DIContainer()
    return _container
```

In Python, if you want to **modify** a global variable inside a function, you must declare `global name` first. Otherwise, assignment creates a local variable.

```python
# Without global
x = 10
def func():
    x = 20  # Creates local x, doesn't change global
    
func()
print(x)  # 10 (global unchanged)

# With global
x = 10
def func():
    global x
    x = 20  # Modifies global
    
func()
print(x)  # 20 (global changed)
```

### 3. **Lambda Functions — Inline Function Creation**

```python
self.register_factory(
    ConfigurationManager,
    lambda: ConfigurationManager()  # Lambda creates function inline
)
```

A **lambda** is a short function in one line. Syntax: `lambda arguments: expression`. Examples:
```python
lambda x: x * 2          # Function that returns x*2
lambda x, y: x + y       # Function with multiple args
lambda: MyClass()        # No args, just create and return instance
```

Lambdas are useful for short, one-off functions. For longer logic, use `def`.

### 4. **`__name__` — Class/Object Name as String**

```python
service_name = interface.__name__
# ConfigurationManager -> "ConfigurationManager"
# TokenManager -> "TokenManager"
```

`__name__` is an attribute containing the object's name as a string. Useful for dictionaries where the key is the type name:
```python
self._services[interface.__name__] = ('singleton', implementation)
# Stores under string key: "ConfigurationManager" -> (singleton impl)
```

### 5. **Dictionary for Service Registry**

```python
self._services: Dict[str, Any] = {}
# Key: service name (string)
# Value: (type, impl) tuple or instance

self._singletons: Dict[str, Any] = {}
# Key: service name (string)
# Value: created instance
```

`_services` stores service **definitions** (not yet created), while `_singletons` stores **created instances** (cached singletons).

### 6. **Checking Existence and Retrieving with `.get()`**

```python
if service_name in self._singletons:
    return self._singletons[service_name]  # Return cached instance
    
if service_name not in self._services:
    raise ValueError(f"Service {service_name} not registered")

service_type, service_impl = self._services[service_name]
```

- `if key in dict:` — check if key exists
- `self._singletons[service_name]` — access value (raises if missing)
- `.get(key, default)` — safely access with default if missing

### 7. **Optional Type (`Optional[DIContainer]`)**

```python
_container: Optional[DIContainer] = None
```

`Optional[DIContainer]` means "either a `DIContainer` instance or `None`." Used when a value might not exist yet (like before the container is created).

### 8. **Tuple Unpacking**

```python
service_type, service_impl = self._services[service_name]
# _services[name] = ('singleton', ConfigurationManager)
# Unpacks into: service_type = 'singleton', service_impl = ConfigurationManager
```

Tuples can be "unpacked"—split into individual variables. Useful when functions return multiple values.

### 9. **Conditional Expression (Ternary)**

```python
if service_type == 'singleton':
    instance = service_impl()  # Call class to instantiate
    self._singletons[service_name] = instance
    return instance
elif service_type == 'factory':
    return service_impl()  # Call factory function
```

Different actions based on `service_type`. If it's 'singleton', instantiate once and cache. If 'factory', call every time.

### 10. **Exception Raising with Context**

```python
raise ValueError(f"Service {service_name} not registered")
```

Raise an exception with a helpful error message. The f-string includes the service name so developers know what was requested.

---

## How It Connects to Other Files

**Imports from:**
- `interfaces.py` — ITokenManager, IUserInteraction (for type hints)
- `services.py` — ConfigurationManager, ApiKeyValidator, ConsoleUserInteraction, LoggingService
- `token_utils.py` — TokenManager
- `models/model_factory.py` — ModelFactory

**Imported by:**
- `main.py` — calls `get_container()` to bootstrap the app
- `utils.py` — calls `get_container()` in convenience functions
- Tests — calls `reset_container()` to inject mocks

**Flow:**
1. **Application startup:**
   - `main.py` calls `get_container()`
   - Container is created and `_setup_default_services()` runs
   - All services are registered; none are created yet (except singletons on first use)

2. **Request for service:**
   - `main.py` calls `container.get_model_factory()`
   - Container checks if `ModelFactory` is registered → yes
   - Container checks if `ModelFactory` singleton already created → no
   - Container sees it's a factory (not singleton), calls the lambda
   - Lambda calls `self.get(ConfigurationManager)`, `self.get(ApiKeyValidator)`, etc.
   - Those services are created on demand (or returned from singleton cache)
   - ModelFactory is constructed with all deps and returned

3. **Testing:**
   - Test calls `reset_container()` → clears global _container
   - Test calls `get_container()` → creates new fresh one
   - Test calls `container.register_instance(IUserInteraction, MockClass())`
   - Now any code asking for user interaction gets the mock

**Key insight:** The container is the **seam** for testing. Instead of modifying code under test, you inject mocks via the container.

---

## Best Practices

**Do:**
- Call `get_container()` at application entry point (main.py)
- Use `reset_container()` in test setup
- Register mocks via `register_instance()` in tests
- Use type hints like `Type[T]` for interface parameters

**Don't:**
- Create DIContainer directly in application code (use `get_container()`)
- Modify `_services` or `_singletons` directly (use register methods)
- Rely on service creation order in `_setup_default_services()` without updating dependencies
- Register the same service twice without resetting first

---

**[← Previous](exceptions.md) | [Back to Index](index.md) | [Next →](services.md)**

*Read next: services.md — understand concrete implementations*
