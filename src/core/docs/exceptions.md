# exceptions.py

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

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `class Child(Parent): pass` | `class Child : Parent()` | All exception classes |
| `super().__init__(message)` | `super(Message)` or init in constructor | LangChainAppError |
| `Optional[str]` | `String?` | error_code, context params |
| `Optional[Dict[str, Any]]` | `Map<String, Any>?` | context field |
| `def __str__(self)` | `override fun toString()` | LangChainAppError |
| `raise SomeError(...)` | `throw SomeError(...)` | All usage sites |
| `except SomeError:` | `catch (e: SomeError)` | Error handling code |
| Multi-level inheritance | `class A : B`, `class B : C`, `class C : Exception` | Exception tree |

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
