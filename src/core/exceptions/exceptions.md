# exceptions.py

**📍 Reading Order:** #3 of 10 core docs | [← Back to Index](index.md) | [Next: dependency_injection.md →](dependency_injection.md)

> Defines a hierarchical custom exception structure for better error handling and debugging throughout the application.

## What This File Does

Python's built-in exceptions are generic (`ValueError`, `RuntimeError`, etc.). This file defines specific exceptions so that code catching errors can distinguish between different failure modes:

- Configuration errors (API key missing, invalid model config)
- Model errors (model failed to initialize, unsupported provider)
- Generation errors (prompt processing failed, token limit exceeded, rate limited)
- Token management errors (token counting failed, cost estimation failed)
- Service errors (DI container failed, service not found)
- User interaction errors (user cancelled, invalid input)
- Strategy errors (unsupported strategy, strategy execution failed)

Each exception category has a **base class** and **specific subclasses**. This lets you catch broad categories (`except GenerationError:`) or specific ones (`except RateLimitError:`).

All inherit from `LangChainAppError` which has `message`, `error_code`, and `context` fields for richer debugging.

---

## Exception Hierarchy Tree

```
Exception
└── LangChainAppError (base for all app errors)
    ├── ConfigurationError (config-related)
    │   ├── ModelConfigurationError
    │   ├── ApiKeyError
    │   └── EnvironmentConfigurationError
    ├── ModelError (model-related)
    │   ├── ModelInitializationError
    │   ├── UnsupportedProviderError
    │   └── ModelValidationError
    ├── GenerationError (generation-related)
    │   ├── PromptError
    │   ├── StreamingError
    │   ├── TokenLimitExceededError
    │   └── RateLimitError
    ├── TokenManagementError (token ops)
    │   ├── TokenCountingError
    │   ├── CostEstimationError
    │   └── UsageTrackingError
    ├── ServiceError (DI / service-related)
    │   ├── DependencyInjectionError
    │   ├── ServiceNotFoundError
    │   └── ServiceInitializationError
    ├── UserInteractionError (user I/O)
    │   ├── UserCancelledError
    │   └── InvalidUserInputError
    └── StrategyError (strategy-related)
        ├── UnsupportedStrategyError
        └── StrategyExecutionError
```

---

## Classes & Exceptions

| Name | Base | Purpose |
|------|------|---------|
| `LangChainAppError` | Exception | Root exception; stores message, error_code, context |
| `ConfigurationError` | LangChainAppError | Base for config issues |
| `ModelConfigurationError` | ConfigurationError | Model config is invalid |
| `ApiKeyError` | ConfigurationError | API key missing/invalid |
| `EnvironmentConfigurationError` | ConfigurationError | Env config broken |
| `ModelError` | LangChainAppError | Base for model issues |
| `ModelInitializationError` | ModelError | Model failed to init |
| `UnsupportedProviderError` | ModelError | Provider not supported |
| `ModelValidationError` | ModelError | Model validation failed |
| `GenerationError` | LangChainAppError | Base for generation issues |
| `PromptError` | GenerationError | Prompt processing failed |
| `StreamingError` | GenerationError | Streaming generation failed |
| `TokenLimitExceededError` | GenerationError | Too many tokens |
| `RateLimitError` | GenerationError | API rate limit hit |
| `TokenManagementError` | LangChainAppError | Base for token ops |
| `TokenCountingError` | TokenManagementError | Token counting failed |
| `CostEstimationError` | TokenManagementError | Cost calc failed |
| `UsageTrackingError` | TokenManagementError | Usage logging failed |
| `ServiceError` | LangChainAppError | Base for service issues |
| `DependencyInjectionError` | ServiceError | DI container error |
| `ServiceNotFoundError` | ServiceError | Service not registered |
| `ServiceInitializationError` | ServiceError | Service init failed |
| `UserInteractionError` | LangChainAppError | Base for user I/O issues |
| `UserCancelledError` | UserInteractionError | User cancelled operation |
| `InvalidUserInputError` | UserInteractionError | User gave bad input |
| `StrategyError` | LangChainAppError | Base for strategy issues |
| `UnsupportedStrategyError` | StrategyError | Strategy not available |
| `StrategyExecutionError` | StrategyError | Strategy failed to run |

---

## LangChainAppError (Detailed)

Root exception for the application.

```python
class LangChainAppError(Exception):
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code  # e.g., "ERR_INVALID_API_KEY"
        self.context = context or {}   # Extra debugging info

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
```

**Usage:**
```python
raise LangChainAppError(
    message="Invalid temperature",
    error_code="ERR_INVALID_TEMP",
    context={"provided": 1.5, "expected": "0.0 to 1.0"}
)
# Prints: [ERR_INVALID_TEMP] Invalid temperature
```

**Kotlin equivalent:**
```kotlin
class LangChainAppError(
    val message: String,
    val errorCode: String? = null,
    val context: Map<String, Any> = emptyMap()
) : Exception(message) {
    override fun toString(): String =
        if (errorCode != null) "[$errorCode] $message" else message
}
```

---

## Exception Categories Explained

### ConfigurationError & Subclasses

Raised when `.env` is missing, API key is invalid, or model config has bad values.

**Who raises it:**
- `ApiKeyValidator.validate_key()` → `ApiKeyError`
- `ClaudeModel._validate_config()` → `ModelConfigurationError`
- Config loading logic → `EnvironmentConfigurationError`

**Catch it to:**
- Inform user to fix `.env` or API key
- Exit gracefully instead of crashing

---

### ModelError & Subclasses

Raised when model can't be created, initialized, or doesn't exist.

**Who raises it:**
- `ModelFactory.create_model()` → `UnsupportedProviderError` if provider unknown
- `ClaudeModel.__init__()` → `ModelInitializationError` if LangChain fails
- Custom validators → `ModelValidationError`

**Catch it to:**
- Log and show user a helpful message
- Fall back to default model
- Retry with different config

---

### GenerationError & Subclasses

Raised during text generation (the core operation).

**Who raises it:**
- `StandardGenerationStrategy.generate()` → `GenerationError` (broad)
- Token manager → `TokenLimitExceededError` if tokens exceed limit
- API → `RateLimitError` when rate limited
- Prompt processing → `PromptError`

**Catch it to:**
- Show user that generation failed
- Suggest retry or suggest shorter input

---

### TokenManagementError & Subclasses

Raised by token counting, cost estimation, or usage logging.

**Who raises it:**
- `TokenManager.count_tokens()` → `TokenCountingError`
- `TokenManager.estimate_cost()` → `CostEstimationError`
- `TokenManager.log_usage()` → `UsageTrackingError`

**Catch it to:**
- Warn user but let generation continue (tokens are secondary)
- Log for debugging

---

### ServiceError & Subclasses

Raised by DI container or service initialization.

**Who raises it:**
- `DIContainer.get()` → `ServiceNotFoundError` if not registered
- Service `__init__()` → `ServiceInitializationError`

**Catch it to:**
- Fix DI configuration
- Restart container

---

### UserInteractionError & Subclasses

Raised by user prompt/input handling.

**Who raises it:**
- `ConsoleUserInteraction.prompt_choice()` → `InvalidUserInputError` on bad choice
- User cancels → `UserCancelledError`

**Catch it to:**
- Reprompt user
- Skip operation

---

### StrategyError & Subclasses

Raised by generation strategies.

**Who raises it:**
- `StandardGenerationStrategy.generate()` → `StrategyExecutionError` on failure
- Strategy selection → `UnsupportedStrategyError` if strategy not available

**Catch it to:**
- Fall back to standard strategy
- Log for debugging

---

## Key Python Concepts (This File)

### 1. **Exception Classes and Inheritance**

```python
class LangChainAppError(Exception):
    """Root exception for all app errors"""
    pass

class ConfigurationError(LangChainAppError):
    """Base for config-related errors"""
    pass

class ApiKeyError(ConfigurationError):
    """API key is missing or invalid"""
    pass
```

Python exceptions inherit from `Exception` (or a subclass). When you create custom exceptions:
- **Root exception** (`LangChainAppError`) inherits from `Exception`
- **Category exceptions** (`ConfigurationError`) inherit from root
- **Specific exceptions** (`ApiKeyError`) inherit from category

This creates a **hierarchy** so you can catch broadly or specifically:
```python
try:
    something()
except ApiKeyError:  # Catch only ApiKeyError
    print("API key problem")
except ConfigurationError:  # Catches ApiKeyError too (it's a subclass)
    print("Some config problem")
except LangChainAppError:  # Catches all app errors
    print("Something went wrong")
except Exception:  # Catches absolutely everything
    print("Unknown error")
```

### 2. **`super().__init__()` — Calling Parent Constructor**

```python
class LangChainAppError(Exception):
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message)  # Call Exception's __init__
        self.message = message
        self.error_code = error_code
        self.context = context or {}
```

`super().__init__(message)` calls the parent class's constructor. This ensures `Exception` is properly initialized with the message. Without it, the exception might not print correctly.

### 3. **`def __str__()` — Customize Exception Output**

```python
def __str__(self) -> str:
    if self.error_code:
        return f"[{self.error_code}] {self.message}"
    return self.message
```

When you print an exception or use `str(exc)`, Python calls `__str__()`. Override it to customize output:
```python
exc = LangChainAppError("Invalid config", error_code="ERR_CONFIG")
print(exc)  # Prints: [ERR_CONFIG] Invalid config
```

### 4. **Raising Exceptions with Context**

```python
raise ApiKeyError(
    message="API key missing",
    error_code="ERR_API_KEY_MISSING",
    context={"provider": "anthropic", "required": True}
)
```

You can pass custom fields to exceptions. The `context` dict lets you attach debugging info (what provider, what value was wrong, etc.) without overcomplicating the exception class.

### 5. **Catching Specific Exception Types**

```python
try:
    create_model("unknown-provider", config)
except UnsupportedProviderError as e:
    print(f"Provider error: {e.message}")
    print(f"Available: {e.context.get('available_providers')}")
except ModelError as e:
    # Catches ModelInitializationError, ModelValidationError, etc.
    print(f"Model problem: {e}")
except LangChainAppError as e:
    # Catch-all for any app exception
    print(f"App error: {e}")
```

Catch specific exception types to handle different errors differently.

### 6. **Exception Hierarchy Benefits**

By organizing exceptions in a hierarchy, you:
- **Avoid generic exceptions**: `except Exception:` catches everything (bad for debugging)
- **Catch what you care about**: `except ApiKeyError:` is specific
- **Catch categories**: `except ModelError:` catches all model-related errors
- **Provide context**: Each exception can have rich `context` for debugging

---

**[← Previous](interfaces.md) | [Back to Index](index.md) | [Next →](dependency_injection.md)**

---

## How It Connects to Other Files

**Imports from:** Standard library only (`typing`, `dataclasses`)

**Imported by:**
- `services.py` — raises `ApiKeyError`
- `dependency_injection.py` — raises `DependencyInjectionError`, `ServiceNotFoundError`
- `models/claude_model.py` — raises `ModelConfigurationError`, `GenerationError`
- `models/model_factory.py` — raises `UnsupportedProviderError`
- `strategies/` — raise `GenerationError`, `StrategyExecutionError`
- `token_utils.py` — could raise token-related errors (currently doesn't, but can add)
- Any file that needs to signal an error

**Flow:**
1. Model creation fails → `ModelFactory` raises `UnsupportedProviderError`
2. User catches it and logs/displays to console
3. Error traceback includes `error_code` and `context` for easier debugging
4. Test can catch specific exception type and verify error handling

**Best practice:**
```python
try:
    model = factory.create_model("openai", config)
except UnsupportedProviderError as e:
    print(f"Provider not available: {e.message}")
    print(f"Available: {e.context.get('available_providers')}")
except LangChainAppError as e:
    # Catch-all for any app error
    print(f"Error: {e}")
```
