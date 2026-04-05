# models/model_factory.md

**📍 Reading Order:** #7 of 10 core docs | [← Back to Index](../index.md) | [Next: claude_model.md →](claude_model.md)

> Factory pattern implementation for creating language models. Handles provider registry, API key validation, and dependency injection wiring.

## What This File Does

The Factory Pattern decouples model creation from usage. Instead of:

```python
# Bad: coupled to ClaudeModel
model = ClaudeModel(config, ...)
```

Use a factory:

```python
# Good: factory handles which class to instantiate
model = factory.create_model("anthropic", config)
# Later: add new provider → just register in factory, no other code changes
```

This file provides `ModelFactory` which:
1. Maintains a **registry** of available models (provider name → class)
2. Validates API keys before creating models
3. Wires dependencies (token manager, user interaction, logging) into created models
4. Supports dynamic registration of new providers (Open/Closed Principle)

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `ModelFactory` | class | Factory for creating model instances |
| `.__init__(config_manager, validator, ...)` | method | Initialize with DI-injected services |
| `.create_model(provider, config)` | method | Create a model for the specified provider |
| `.get_available_providers()` | method | List all registered providers |
| `.register_model(provider, model_class)` | method | Register a new provider (extensibility) |
| `.create_default_claude_model(model_name)` | method | Convenience for quick Claude creation |

---

## ModelFactory Class (Detailed)

### __init__

```python
def __init__(
    self,
    config_manager: ConfigurationManager,
    api_key_validator: IApiKeyValidator,
    token_manager: ITokenManager,
    user_interaction: IUserInteraction,
    logging_service
):
    """Initialize the model factory with all dependencies."""
    self.config_manager = config_manager
    self.api_key_validator = api_key_validator
    self.token_manager = token_manager
    self.user_interaction = user_interaction
    self.logging_service = logging_service
    self.logger = logging_service.get_logger(__name__)

    # Registry of available model classes
    self._model_registry: Dict[str, Type[ILanguageModel]] = {
        "anthropic": ClaudeModel,
    }
```

**Registry:** Dictionary where keys are provider names ("anthropic") and values are **class objects** (not instances).

**Type hint:** `Type[ILanguageModel]` means "a class object that implements ILanguageModel" (not an instance of one).

**Kotlin equivalent:**
```kotlin
class ModelFactory(
    configManager: ConfigurationManager,
    apiKeyValidator: IApiKeyValidator,
    ...
) {
    private val modelRegistry: Map<String, KClass<out ILanguageModel>> = mapOf(
        "anthropic" to ClaudeModel::class
    )
}
```

### create_model()

```python
def create_model(self, provider: str, config: ModelConfig) -> ILanguageModel:
    """
    Create a model instance for the specified provider.

    Args:
        provider: The model provider (anthropic, etc.)
        config: Model configuration

    Returns:
        Configured model instance

    Raises:
        UnsupportedProviderError: If provider not supported
        ApiKeyError: If API key validation fails
    """
    provider_lower = provider.lower()

    # 1. Check if provider is supported
    if provider_lower not in self._model_registry:
        available = ", ".join(self._model_registry.keys())
        raise UnsupportedProviderError(
            f"Provider '{provider}' not supported. Available providers: {available}"
        )

    # 2. Get API key (from config or environment)
    if not config.api_key:
        config.api_key = self.config_manager.get_api_key(provider_lower)

    # 3. Validate API key
    self.api_key_validator.validate_key(config.api_key, provider_lower)

    # 4. Get model class from registry
    model_class = self._model_registry[provider_lower]
    logger = self.logging_service.get_logger(f"{__name__}.{provider_lower}")

    self.logger.info(f"Creating {provider} model: {config.model_name}")

    # 5. Instantiate with DI-injected dependencies
    return model_class(
        config=config,
        token_manager=self.token_manager,
        user_interaction=self.user_interaction,
        logger=logger
    )
```

**Flow:**
1. Normalize provider name to lowercase
2. Check registry: does provider exist?
3. If config doesn't have API key, get it from environment
4. Validate API key (fails if missing or invalid)
5. Look up model class in registry
6. Create logger with provider-specific tag
7. Instantiate model class with all DI dependencies
8. Return configured model

**Key:** All dependencies (token_manager, user_interaction, logger) are injected—the factory doesn't create them, it passes them to the model.

### get_available_providers()

```python
def get_available_providers(self) -> list[str]:
    """Get list of available providers."""
    return list(self._model_registry.keys())
```

Returns ["anthropic"] currently, but can include more if registered.

### register_model()

```python
def register_model(self, provider: str, model_class: Type[ILanguageModel]) -> None:
    """
    Register a new model class.

    Args:
        provider: Provider name
        model_class: Model class to register
    """
    self._model_registry[provider.lower()] = model_class
    self.logger.info(f"Registered model provider: {provider}")
```

Allows adding new providers **without modifying** the factory code (Open/Closed Principle).

**Example:**
```python
class OpenAIModel(ILanguageModel):
    # ... implementation ...
    pass

factory.register_model("openai", OpenAIModel)
# Now factory.create_model("openai", config) works
```

### create_default_claude_model()

```python
def create_default_claude_model(self, model_name: str = "claude-3-haiku-20240307") -> ILanguageModel:
    """Create a default Claude model with standard configuration."""
    config = ModelConfig(
        model_name=model_name,
        temperature=0.2,
        max_tokens=512
    )
    return self.create_model("anthropic", config)
```

Convenience method for quick model creation with sensible defaults.

---

## Registry Pattern

The factory uses a **registry** — a data structure that maps names to implementations.

```python
self._model_registry: Dict[str, Type[ILanguageModel]] = {
    "anthropic": ClaudeModel,
    # "openai": OpenAIModel,  # Could add more
    # "huggingface": LocalHFModel,  # Could add more
}
```

**Advantages:**
- **Open/Closed:** Add new models without modifying create_model()
- **Loose coupling:** Code using models never knows about specific classes
- **Testability:** Register mock models for testing

**Kotlin equivalent:**
```kotlin
private val modelRegistry = mutableMapOf<String, KClass<out ILanguageModel>>(
    "anthropic" to ClaudeModel::class
)

fun registerModel(provider: String, modelClass: KClass<out ILanguageModel>) {
    modelRegistry[provider.lowercase()] = modelClass
}
```

---

## Dependency Injection in Factory

The factory is a **service locator** pattern—it knows how to construct and wire services:

```python
def create_model(self, provider: str, config: ModelConfig) -> ILanguageModel:
    # The factory has these injected
    model_class = self._model_registry[provider_lower]
    token_manager = self.token_manager  # From constructor
    logger = self.logging_service.get_logger(...)  # From constructor

    # The factory passes them to the model
    return model_class(
        config=config,
        token_manager=token_manager,
        user_interaction=self.user_interaction,
        logger=logger
    )
```

**Why inject into factory?** So the factory doesn't create services (config manager, token manager, etc.), but receives them. This means:
- Easy to test (inject test doubles)
- Single source of wiring (DI container sets up factory once)
- Services are shared (same token_manager instance across all models)

---

## Key Python Concepts (This File)

### 1. **Type Hints with Class Objects (`Type[ILanguageModel]`)**

```python
self._model_registry: Dict[str, Type[ILanguageModel]] = {
    "anthropic": ClaudeModel,  # Store class objects, not instances
}
```

`Type[ILanguageModel]` means "a class object that implements ILanguageModel" (not an instance). So:

- `Type[ILanguageModel]` = the class itself: `ClaudeModel`, `OpenAIModel`, etc.
- `ILanguageModel` = an instance of the class: `ClaudeModel(...)`

When you store `ClaudeModel` in the registry, you later instantiate it:
```python
model_class = self._model_registry[provider_lower]  # Get the class
instance = model_class(config, ...)  # Call it to create instance
```

This is how **factories** work—they store class blueprints and instantiate them on demand.

### 2. **Dictionary with Class Values**

```python
Dict[str, Type[ILanguageModel]]  # "anthropic" → ClaudeModel class
```

Regular dictionary is `Dict[str, str]` (keys and values are both strings). This one maps strings to **class objects**. It's a registry pattern—you look up the provider name and get the class to instantiate.

### 3. **`provider.lower()` — String Normalization**

```python
provider_lower = provider.lower()
if provider_lower not in self._model_registry:
    # ...
```

`lower()` converts to lowercase: `"Anthropic"` → `"anthropic"`. This makes lookups case-insensitive so "ANTHROPIC", "Anthropic", and "anthropic" all work.

### 4. **`"key" in dict` and `"key" not in dict`**

```python
if provider_lower not in self._model_registry:
    raise UnsupportedProviderError(...)
```

- `key in dict` → `True` if key exists, `False` otherwise
- `key not in dict` → opposite

Unlike `.get()` which returns a value, `in` just checks existence.

### 5. **`", ".join(list)` — Join Strings with Separator**

```python
available = ", ".join(self._model_registry.keys())
# If keys are ["anthropic", "openai"], result is "anthropic, openai"
```

`", ".join(sequence)` combines list items into one string with `", "` between them. Useful for error messages like:
```
"Available providers: anthropic, openai"
```

### 6. **`if not value:` — Falsy Checks**

```python
if not config.api_key:
    config.api_key = self.config_manager.get_api_key(provider_lower)
```

In Python, many values are **falsy**: `None`, `""` (empty string), `0`, `[]` (empty list), etc.

`if not config.api_key:` checks if the API key is falsy (missing, empty, or None). If so, fetch it from config.

### 7. **Logging with Context (`f"...{variable}..."`)**

```python
self.logger.info(f"Creating {provider} model: {config.model_name}")
```

`f"..."` (f-string) embeds variables in strings:
- `f"Creating {provider} model"` becomes `"Creating anthropic model"` if `provider = "anthropic"`
- Easier than string concatenation: `"Creating " + provider + " model"`

### 8. **Modifying Config Objects**

```python
if not config.api_key:
    config.api_key = self.config_manager.get_api_key(provider_lower)
```

`config` is an object (likely a `@dataclass`). You can assign to its fields to modify it. This mutates the config in-place; if the caller has a reference to it, they see the change.

---

## How It Connects to Other Files

**Imports from:**
- `interfaces.py` — ILanguageModel, ITokenManager, IUserInteraction, IApiKeyValidator, ModelConfig
- `exceptions.py` — UnsupportedProviderError
- `services.py` — ConfigurationManager, ApiKeyValidator
- `models/claude_model.py` — ClaudeModel (the model class)

**Imported by:**
- `dependency_injection.py` — factory registered in DI container
- Any code that needs to create models (delegates to container)

**Flow:**
1. **DI container initialization:**
   - Creates ConfigurationManager
   - Creates ApiKeyValidator
   - Creates TokenManager
   - Creates ConsoleUserInteraction
   - Creates LoggingService
   - Creates ModelFactory with all above (via lambda factory)

2. **Model creation:**
   - User code calls `container.get_model_factory()`
   - Gets ModelFactory instance
   - Calls `factory.create_model("anthropic", config)`
   - Factory validates API key, looks up ClaudeModel in registry
   - Factory creates ClaudeModel with injected services
   - Returns configured model

3. **Model generation:**
   - User calls `model.generate(prompt)`
   - Model uses token_manager (from injection) for cost tracking
   - Model uses user_interaction (from injection) for prompts
   - Model uses logger (from injection) for logging

---

## Design Principles

**Open/Closed Principle:**
- Open for extension: `factory.register_model("openai", OpenAIModel)`
- Closed for modification: `create_model()` code doesn't change

**Dependency Inversion:**
- Factory depends on interfaces (ILanguageModel, ITokenManager), not concrete classes
- Models depend on interfaces, not the factory

**Single Responsibility:**
- Factory: create models and wire dependencies
- Models: generate text
- Validators: validate API keys
- Token manager: track tokens
- User interaction: handle I/O

---

## Examples

**Creating a Claude model:**
```python
factory = container.get_model_factory()
config = ModelConfig(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.5,
    max_tokens=1024
)
model = factory.create_model("anthropic", config)
result = model.generate("Hello!")
print(result.content)
```

**Registering a new provider (extensibility):**
```python
class MyCustomModel(ILanguageModel):
    def _validate_config(self) -> None: ...
    def generate(self, prompt: str, **kwargs) -> GenerationResult: ...
    def get_model_info(self) -> Dict[str, Any]: ...
    @property
    def provider(self) -> str:
        return "MyCustom"

factory.register_model("mycustom", MyCustomModel)
config = ModelConfig(model_name="my-model")
model = factory.create_model("mycustom", config)
# Works! No changes to create_model() code
```

---

**[← Previous](../utils.md) | [Back to Index](../index.md) | [Next →](claude_model.md)**

*Read next: claude_model.md — see a specific model implementation*
