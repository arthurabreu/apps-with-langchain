# dependency_injection.md

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

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `TypeVar('T')` | `inline fun <reified T>` | Type parameter T |
| `Type[T]` (class object) | `KClass<T>` or just `T::class` | interface parameter in register methods |
| `Dict[str, Any]` | `Map<String, Any>` | _services, _singletons |
| `Optional[DIContainer]` | `DIContainer?` | _container global |
| `global _container` | Not needed; Kotlin uses `object` or `companion object` | Module-level singleton pattern |
| `lambda: MyClass()` | `{ MyClass() }` lambda | Factory functions in register_factory |
| `service_impl()` | `serviceImpl.invoke()` or just `serviceImpl()` | In get() when instantiating |
| `__name__` attribute | `::class.simpleName` | Getting service name for registry key |
| `.get()` dict method | `.get(key)` or `[key]?` | Retrieving from _services, _singletons |
| `.get(key, default)` | `[key] ?: default` | In get() method fallback |
| `raise ValueError()` | `throw IllegalArgumentException()` | Error handling in get() |
| `if ... in dict:` | `if (key in map)` or `map.containsKey(key)` | Checking if service registered |

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
