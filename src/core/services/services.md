# services.py

**📍 Reading Order:** #5 of 10 core docs | [← Back to Index](index.md) | [Next: utils.md →](utils.md)

> Contains concrete implementations of core services: config management, user interaction, API validation, and logging.

## What This File Does

This file provides **concrete implementations** of the interfaces defined in `interfaces.py`. Services are stateful objects that provide specific capabilities:

- **ConfigurationManager** — reads environment variables, provides config values
- **ApiKeyValidator** — checks API keys are valid before using them
- **ConsoleUserInteraction** — prompts user on console, displays messages
- **LoggingService** — sets up Python logging with file + console handlers

Think of this as the **implemention layer** — where interfaces meet reality (console I/O, environment, logging).

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `ApiKeyValidator` | class | Validates API keys for different providers |
| `.validate_key(api_key, provider)` | method | Check if API key looks valid for provider |
| `ConsoleUserInteraction` | class | Prompts user on console, displays output |
| `.prompt_continue()` | method | Ask user to press Enter or 's' to skip |
| `.prompt_choice(message, choices)` | method | Display numbered choices, get user selection |
| `.display_info(message)` | method | Print info message to console + log |
| `.display_error(error)` | method | Print error message to console + log |
| `.display_warning(message)` | method | Print warning message to console + log |
| `ConfigurationManager` | class | Load and access config from environment |
| `.get(key, default)` | method | Get config value |
| `.set(key, value)` | method | Set config value (in-memory only) |
| `.get_api_key(provider)` | method | Get API key for a specific provider |
| `LoggingService` | class | Setup and provide logger instances |
| `.get_logger(name)` | method | Get a logger for a module |

---

## ApiKeyValidator (Detailed)

Validates API keys before they're used in model creation.

```python
class ApiKeyValidator:
    def validate_key(self, api_key: str, provider: str) -> bool:
        """
        Validate API key format and content.

        Args:
            api_key: The key string to validate
            provider: "anthropic", "huggingface", etc.

        Returns:
            True if valid

        Raises:
            ApiKeyError: If key is missing or obviously invalid
        """
        # Checks:
        # 1. Key is not empty
        # 2. Key doesn't contain placeholder strings like "your-api-key"
        # 3. Key passes provider-specific format checks
        #    - Anthropic: >= 20 chars
        #    - HuggingFace: starts with "hf_"
```

**Example:**
```python
validator = ApiKeyValidator()
validator.validate_key("sk-...", "anthropic")  # OK
validator.validate_key("your-api-key", "anthropic")  # Raises ApiKeyError
```

**Flow:** Used by `ModelFactory.create_model()` before creating a model instance.

---

## ConsoleUserInteraction (Detailed)

Handles all user prompts and output on the console.

```python
class ConsoleUserInteraction:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
```

| Method | Purpose |
|--------|---------|
| `prompt_continue()` | Loop until user presses Enter (continue) or 's' (skip); return bool |
| `prompt_choice(message, choices)` | Display message + numbered choices; return selected string |
| `display_info(message)` | Print `[INFO]` prefix + log to logger |
| `display_error(error)` | Print `[ERROR]` prefix + log to logger |
| `display_warning(message)` | Print `[WARNING]` prefix + log to logger |

**Example:**
```python
ui = ConsoleUserInteraction()
choice = ui.prompt_choice(
    "Which model?",
    ["Claude", "Local HF", "Other"]
)  # User sees:
# Which model?
# 1. Claude
# 2. Local HF
# 3. Other
# Enter your choice (number): _

ui.display_info("Starting generation...")  # Prints [INFO] and logs
```

**Note:** All display methods both print to console *and* log (if logger available). This ensures messages appear to user and are recorded in logs.

---

## ConfigurationManager (Detailed)

Loads configuration from environment variables (typically from `.env`).

```python
class ConfigurationManager:
    def __init__(self):
        self._config = {}
        self._load_from_env()

    def _load_from_env(self) -> None:
        # Reads from environment:
        # ANTHROPIC_API_KEY
        # HUGGINGFACE_API_KEY
        # ENVIRONMENT (default: "development")
        # LOG_LEVEL (default: "INFO")
```

| Method | What It Does |
|--------|-------------|
| `get(key, default=None)` | Get config value or default |
| `set(key, value)` | Set config in memory (not persisted to .env) |
| `get_api_key(provider)` | Get API key for "anthropic" or "huggingface" |

**Example:**
```python
config = ConfigurationManager()
api_key = config.get_api_key("anthropic")  # Gets ANTHROPIC_API_KEY from env
env = config.get("environment", "development")
```

**Key point:** This is a simple **facade** over `os.getenv()`. It doesn't write back to `.env`; changes are in-memory only.

---

## LoggingService (Detailed)

Sets up Python's logging system with both console and file output.

```python
class LoggingService:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._setup_logging()

    def _setup_logging(self) -> None:
        # Configure logging based on LOG_LEVEL from config
        # Creates handlers:
        # 1. StreamHandler (console output)
        # 2. FileHandler (writes to langchain_app.log)
```

| Method | What It Does |
|--------|-------------|
| `get_logger(name)` | Get a logger for a module (e.g., "services.py") |

**Example:**
```python
logging_service = LoggingService(config_manager)
logger = logging_service.get_logger("models.claude")
logger.info("Claude model initialized")
# Output: 2026-03-13 14:25:30,123 - models.claude - INFO - Claude model initialized
```

**Note:** Python's `logging.getLogger(name)` returns a logger with that name. You can call it multiple times with same name and get the same logger. The service just wraps it.

---

## Key Python Concepts (This File)

### 1. **`__init__` and Instance Attributes (self.variable)**

The `__init__` method is Python's constructor. When you write:

```python
class ConfigurationManager:
    def __init__(self):
        self._config = {}  # Create instance variable
        self._load_from_env()
```

The line `self._config = {}` creates an **instance attribute**—a variable that belongs to this specific object. Every instance of `ConfigurationManager` has its own `_config` dictionary. The underscore prefix (`_config` instead of `config`) is a Python convention meaning "private or internal—don't access this directly from outside the class."

Why use `self.variable = {}`? Because then all methods in the class can access it via `self.variable`. Without storing it on `self`, it would be a local variable that disappears when `__init__` finishes.

### 2. **Default Parameters (`logger: logging.Logger = None`)**

```python
def __init__(self, logger: logging.Logger = None):
    self.logger = logger or logging.getLogger(__name__)
```

The `= None` means "if the caller doesn't provide `logger`, default to `None`." The line `logger or logging.getLogger(__name__)` uses Python's `or` operator: if `logger` is `None` (falsy), create a new logger. This pattern lets you optionally inject a logger for testing, or use a default one if you don't provide one.

### 3. **Underscore Prefixes in Python**

- **`_private_var`** (single underscore) — "This is internal; don't use it from outside the class." It's a **convention**, not enforced. Python still lets you access it, but you're breaking a contract.
- **`__dunder_var`** (double underscore) — **Name mangling** applies. Python actually renames this to `_ClassName__dunder_var` to make accidental override harder. Rarely used; single underscore is preferred.
- **`regular_var`** — Public; intended for external use.

In this file, `_config` and `_load_from_env()` both use single underscores to signal they're implementation details, not part of the public API.

### 4. **Class Methods vs Instance Methods**

All methods in this file are **instance methods** (operate on one object):

```python
class ConfigurationManager:
    def __init__(self):
        self._config = {}
    
    def get(self, key, default=None):
        return self._config.get(key, default)  # Uses self._config
```

When you call `config.get("key")`, Python automatically passes `config` as `self`. You never explicitly write `self` in the call.

### 5. **Dictionary `.get()` Method**

```python
value = dict.get(key, default_value)
```

If `key` exists, return its value. If not, return `default_value` (or `None` if no default given). This is safer than `dict[key]` which raises `KeyError` if the key is missing.

### 6. **Logging Setup (`logging.basicConfig`, handlers)**

```python
def _setup_logging(self) -> None:
    logging.basicConfig(level=log_level)
    handler = logging.FileHandler("app.log")
    # ...
```

Python's `logging` module manages **handlers** — destinations where log messages go. `StreamHandler` sends to console, `FileHandler` sends to a file. You can attach multiple handlers to the same logger so messages go to both console and file simultaneously.

### 7. **`@property` (Read-Only Fields)**

Some classes expose attributes as properties:

```python
@property
def logger(self):
    return self._logger
```

This makes `obj.logger` readable like a field, but it's actually calling a method. Useful for computed values or controlled access.

---

## How It Connects to Other Files

**Imports from:** Standard library (`os`, `logging`, `typing`)

**Imported by:**
- `dependency_injection.py` — registers these services in the DI container
- `models/claude_model.py` — uses logger
- `models/model_factory.py` — uses all: config, validator, logging

**Flow:**
1. **DI container initialization:**
   - Creates `ConfigurationManager` (singleton)
   - Creates `LoggingService` (depends on ConfigManager)
   - Creates `ApiKeyValidator` (singleton)
   - Creates `ConsoleUserInteraction` (depends on LoggingService)

2. **Model creation:**
   - `ModelFactory.create_model()` calls `config_manager.get_api_key()`
   - Then calls `api_key_validator.validate_key()`
   - If valid, creates model with logger from `logging_service.get_logger()`

3. **User interaction:**
   - `ClaudeModel.generate()` calls `user_interaction.prompt_continue()`
   - Strategies call `user_interaction.display_info()`, etc.

4. **Logging:**
   - Every component gets a logger from `LoggingService`
   - All logs go to both console (via `print()` in display methods) and file

**Key insight:** These are **concrete implementations** of Protocols from `interfaces.py`. They're injected via the DI container so they can be easily mocked in tests.

---

**[← Previous](dependency_injection.md) | [Back to Index](index.md) | [Next →](utils.md)**

*Read next: utils.md — understand convenience functions*
