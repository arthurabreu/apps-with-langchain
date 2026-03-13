# models/claude_model.py

> Claude language model implementation. Extends ILanguageModel (ABC) and integrates with strategies, token management, and user interaction.

## What This File Does

This file provides a **concrete implementation** of `ILanguageModel` for Anthropic's Claude models. It:

1. Wraps LangChain's `ChatAnthropic` for Claude API access
2. Validates configuration (API key, temperature, max tokens)
3. Initializes the underlying LangChain model
4. Delegates text generation to pluggable **strategies** (standard vs. streaming)
5. Integrates with `TokenManager` for cost tracking and `UserInteraction` for prompts

The class is heavily dependent on DI—it receives token_manager, user_interaction, and logger as constructor arguments, not creating them itself.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `_STRATEGIES` | dict | Maps GenerationStrategy enum to strategy class |
| `ClaudeModel` | class | Concrete ILanguageModel implementation for Claude |
| `.__init__(config, token_manager, ...)` | method | Initialize with DI-injected dependencies |
| `._validate_config()` | method | Check config is valid (overrides ABC method) |
| `._initialize_model()` | method | Create LangChain ChatAnthropic instance |
| `.generate(prompt, **kwargs)` | method | Generate text (overrides ABC method) |
| `.get_model_info()` | method | Return config details (overrides ABC method) |
| `.provider` | @property | Return "Anthropic" (overrides ABC property) |

---

## Module-Level Registry (_STRATEGIES)

```python
_STRATEGIES = {
    GenerationStrategy.STANDARD: StandardGenerationStrategy,
    GenerationStrategy.STREAMING: StreamingGenerationStrategy,
}
```

Maps enum to **class objects** (not instances). When generating, the code:
1. Looks up the strategy class: `strategy_class = _STRATEGIES[self.config.generation_strategy]`
2. Instantiates it: `strategy = strategy_class(token_manager, user_interaction, logger)`
3. Calls its generate() method

This is the **Strategy Pattern**: swap behavior without changing code.

**Kotlin equivalent:**
```kotlin
val STRATEGIES = mapOf(
    GenerationStrategy.STANDARD to StandardGenerationStrategy::class,
    GenerationStrategy.STREAMING to StreamingGenerationStrategy::class
)
```

---

## ClaudeModel Class (Detailed)

### __init__

```python
def __init__(
    self,
    config: ModelConfig,
    token_manager: ITokenManager,
    user_interaction: IUserInteraction,
    logger: logging.Logger
):
    self.token_manager = token_manager
    self.user_interaction = user_interaction
    self.logger = logger
    self._model = None

    super().__init__(config)  # Calls parent's __init__ which calls _validate_config()
    self._initialize_model()
```

**Key point:** Set instance fields **before** calling `super().__init__(config)` because the parent's `__init__`:
1. Calls `self.config = config`
2. Calls `self._validate_config()` (which this class overrides)
3. If `_validate_config()` needs `self.token_manager`, it must already be set

This is why we set fields first, then call super.

### _validate_config()

```python
def _validate_config(self) -> None:
    """Validate model configuration."""
    if not self.config.api_key:
        raise ApiKeyError("Anthropic API key not provided")

    if not self.config.model_name:
        raise ModelConfigurationError("Model name not provided")

    if not (0.0 <= self.config.temperature <= 1.0):
        raise ModelConfigurationError("Temperature must be between 0.0 and 1.0")

    if self.config.max_tokens <= 0:
        raise ModelConfigurationError("Max tokens must be positive")
```

Checks:
- API key provided (not None or empty)
- Model name provided
- Temperature in valid range [0, 1]
- Max tokens > 0

Raises specific exceptions for each case. Called automatically by parent `__init__`.

### _initialize_model()

```python
def _initialize_model(self) -> None:
    """Initialize the Claude model (LangChain ChatAnthropic)."""
    try:
        self.logger.info(f"Initializing Claude model: {self.config.model_name}")

        self._model = ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key
        )

        self.logger.info("Claude model initialized successfully")
    except Exception as e:
        self.logger.error(f"Failed to initialize Claude model: {e}")
        raise ModelConfigurationError(f"Failed to initialize Claude model: {e}")
```

Creates the actual LangChain model. If it fails, wraps the error in a specific exception.

**LangChain's ChatAnthropic:**
- Takes model name, temperature, max_tokens, API key
- Provides `.invoke()` for sync generation
- Provides `.stream()` for streaming

### generate()

```python
def generate(self, prompt: str, **kwargs) -> GenerationResult:
    """
    Generate text from a prompt.

    Args:
        prompt: Input text prompt
        **kwargs: Additional parameters (skip_prompt=True to skip user prompt)

    Returns:
        GenerationResult with content and metadata

    Raises:
        GenerationError: If generation fails
    """
    try:
        # Check if user wants to continue
        skip_prompt = kwargs.get('skip_prompt', False)
        if not skip_prompt and not self.user_interaction.prompt_continue():
            self.user_interaction.display_info("Generation skipped by user.")
            return GenerationResult(content="Generation skipped by user.")

        self.user_interaction.display_info(f"Prompt: {prompt[:100]}...")
        self.user_interaction.display_info("Generating using Claude...")

        # Resolve strategy from registry
        strategy_class = _STRATEGIES.get(self.config.generation_strategy)
        if not strategy_class:
            raise GenerationError(f"Unknown generation strategy: {self.config.generation_strategy}")

        # Create strategy instance with dependencies
        strategy = strategy_class(
            self.token_manager,
            self.user_interaction,
            self.logger
        )

        # Generate using the selected strategy
        result = strategy.generate(self._model, prompt, self.config)

        self.user_interaction.display_info("Generation complete!")
        return result

    except Exception as e:
        error_msg = f"Generation failed: {e}"
        self.logger.error(error_msg)
        self.user_interaction.display_error(error_msg)
        raise GenerationError(error_msg)
```

**Flow:**
1. Ask user if they want to continue (unless skip_prompt=True)
2. Display prompt to user
3. Look up strategy from _STRATEGIES registry
4. Instantiate strategy with DI-injected dependencies
5. Call strategy.generate() to do the actual work
6. Return result
7. On error, log and display to user, then raise

### get_model_info()

```python
def get_model_info(self) -> Dict[str, Any]:
    """Get information about the model."""
    return {
        "provider": self.provider,
        "model_name": self.config.model_name,
        "temperature": self.config.temperature,
        "max_tokens": self.config.max_tokens,
        "status": "ready" if self._model else "not_initialized"
    }
```

Returns a dict of the model's configuration and status.

### provider (Property)

```python
@property
def provider(self) -> str:
    """Get the model provider name."""
    return "Anthropic"
```

Returns provider name. Used by other code to identify model origin.

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `class ClaudeModel(ILanguageModel):` | `class ClaudeModel(ILanguageModel)` | Class definition |
| `super().__init__(config)` | `super(config)` | Calling parent constructor |
| `self._model = None` | `private var _model: ChatAnthropic? = null` | Instance field |
| `def __init__(self, ...)` | `constructor(...)` | Constructor |
| `raise SomeError()` | `throw SomeError()` | Raising exceptions |
| `except Exception as e` | `catch (e: Exception)` | Catching exceptions |
| `.get(key)` dict method | `.get(key)` or `[key]` | Dictionary access |
| `if not value:` | `if (!value) {}` or `if (value == null)` | Negation/null check |
| `@property` | `val` or computed property | Provider property |
| `@abstractmethod` override | `override fun` | Overriding abstract method |
| `Dict[str, Any]` | `Map<String, Any>` | Return type |
| `logger.info()`, `error()` | `Log.i(tag, msg)` or `Timber.i(msg)` | Logging |
| `**kwargs` | Named parameters or `vararg` | Additional parameters |
| `.get(key, default)` | `[key] ?: default` | Dictionary with default |
| `if ... in dict` | `if (key in dict)` or `.containsKey(key)` | Key existence check |

---

## How It Connects to Other Files

**Imports from:**
- `interfaces.py` — ILanguageModel (ABC), ITokenManager, IUserInteraction, ModelConfig, GenerationResult, GenerationStrategy
- `exceptions.py` — ModelConfigurationError, ApiKeyError, GenerationError
- `strategies/` — StandardGenerationStrategy, StreamingGenerationStrategy
- External: `langchain_anthropic.ChatAnthropic`

**Imported by:**
- `models/model_factory.py` — creates instances via factory
- DI container (`dependency_injection.py`) — registered via ModelFactory
- Tests — instantiate directly with mocked dependencies

**Flow:**
1. **Model creation:**
   - `ModelFactory.create_model("anthropic", config)` looks up ClaudeModel in registry
   - Calls `ClaudeModel(config, token_manager, user_interaction, logger)`
   - Constructor validates config, initializes LangChain ChatAnthropic

2. **Generation:**
   - User calls `claude_model.generate(prompt)`
   - Model validates config in `_validate_config()` (via parent init)
   - Model picks strategy from _STRATEGIES based on config.generation_strategy
   - Strategy calls `self._model.invoke()` or `self._model.stream()` (LangChain)
   - Result bubbles back to user

3. **Token tracking:**
   - Strategy calls `self.token_manager.count_tokens()`
   - Strategy calls `self.token_manager.log_usage()`
   - User can inspect costs via `token_manager.get_usage_summary()`

4. **User interaction:**
   - Model calls `user_interaction.prompt_continue()` before generating
   - Strategy displays token analysis via `user_interaction.display_info()`
   - On error, `user_interaction.display_error()`

---

## Key Patterns

**DI constructor pattern:**
```python
def __init__(self, config, token_manager, user_interaction, logger):
    self.token_manager = token_manager  # Don't create, receive it
    # ... set other fields ...
    super().__init__(config)  # Parent validates via _validate_config()
    self._initialize_model()
```

**ABC inheritance and override:**
```python
class ClaudeModel(ILanguageModel):  # Explicit inheritance required (ABC)
    def _validate_config(self) -> None:  # Override abstract method
        # Custom validation
    @property
    def provider(self) -> str:  # Implement abstract property
        return "Anthropic"
```

**Strategy dispatch via registry:**
```python
strategy_class = _STRATEGIES.get(self.config.generation_strategy)
strategy = strategy_class(deps...)  # Instantiate with deps
result = strategy.generate(...)  # Call method
```

**Error wrapping:**
```python
try:
    # Do work
except Exception as e:
    self.logger.error(msg)
    self.user_interaction.display_error(msg)
    raise GenerationError(msg)  # Wrap in domain-specific exception
```
