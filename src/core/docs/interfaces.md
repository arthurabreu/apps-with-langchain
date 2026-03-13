# interfaces.py

**📍 Reading Order:** #2 of 10 core docs | [← Back to Index](index.md) | [Next: exceptions.md →](exceptions.md)

> Defines abstract contracts (ABC, Protocol) and data classes for the entire application.

## What This File Does

This file is the **contract layer**. It declares:

1. **Data classes** (`ModelConfig`, `GenerationResult`) — immutable data structures passed around the system
2. **Enum** (`GenerationStrategy`) — constant set of strategy choices
3. **Protocols** (structural typing) — interfaces checked at type-hint time, not inheritance time
4. **Abstract Base Classes (ABC)** — interfaces that *require* explicit inheritance

Think of Protocols as Kotlin `interface`s where you *don't* write `implements IFoo`, but the type-checker verifies you have the right methods. ABCs are like Kotlin `abstract class` where you *must* explicitly inherit.

### Why Segregation?

Token management is split across 3 protocols (`ITokenCounter`, `ICostEstimator`, `IUsageTracker`) then combined into `ITokenManager`. User interaction split into `IUserPrompt` and `IUserDisplay`, then combined. This is the **Interface Segregation Principle**: clients depend only on what they use.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `GenerationStrategy` | Enum | Choices: `STANDARD`, `STREAMING` |
| `ModelConfig` | @dataclass | Configuration for language models (model_name, temperature, max_tokens, api_key, system_message, generation_strategy) |
| `GenerationResult` | @dataclass | Result of a generation (content, tokens_used, cost, metadata, strategy_used) |
| `ITokenCounter` | Protocol | `count_tokens(text: str, model_name: str) -> int` |
| `ICostEstimator` | Protocol | `estimate_cost(tokens: int, model_name: str, is_output: bool) -> float` |
| `IUsageTracker` | Protocol | `log_usage(...)`, `get_usage_summary() -> Dict` |
| `ITokenManager` | Protocol | Combines all three token protocols |
| `IUserPrompt` | Protocol | `prompt_continue() -> bool`, `prompt_choice(message: str, choices: list[str]) -> str` |
| `IUserDisplay` | Protocol | `display_info(msg)`, `display_error(err)`, `display_warning(msg)` |
| `IUserInteraction` | Protocol | Combines prompt and display |
| `IModelValidator` | Protocol | `validate_config(config: ModelConfig) -> None` |
| `IApiKeyValidator` | Protocol | `validate_key(api_key: str, provider: str) -> bool` |
| `IGenerationStrategy` | Protocol | `generate(model, prompt, config) -> GenerationResult`, `supports_model(provider) -> bool` |
| `ILanguageModel` | ABC | Abstract base; requires `_validate_config()`, `generate()`, `get_model_info()`, `provider` property |
| `IModelFactory` | Protocol | `create_model()`, `get_available_providers()`, `register_model()` |

---

## Detailed Reference

### GenerationStrategy (Enum)

```python
GenerationStrategy.STANDARD  # Sync generation
GenerationStrategy.STREAMING # Real-time streaming output
```

Used in `ModelConfig.generation_strategy` to tell the model which strategy to use.

**Kotlin equivalent:**
```kotlin
enum class GenerationStrategy {
    STANDARD,
    STREAMING
}
```

---

### ModelConfig (@dataclass)

Configuration object passed to model factories and models.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `model_name` | str | — | "claude-3-haiku-20240307" etc. |
| `temperature` | float | 0.2 | 0.0 (deterministic) to 1.0 (random) |
| `max_tokens` | int | 512 | Max output length |
| `api_key` | Optional[str] | None | API key for the provider |
| `system_message` | Optional[str] | None | System prompt / role definition |
| `generation_strategy` | GenerationStrategy | STANDARD | Which strategy to use |
| `additional_params` | Optional[Dict] | None | Extra kwargs to pass to model |

**Kotlin equivalent:**
```kotlin
data class ModelConfig(
    val modelName: String,
    val temperature: Float = 0.2f,
    val maxTokens: Int = 512,
    val apiKey: String? = null,
    val systemMessage: String? = null,
    val generationStrategy: GenerationStrategy = GenerationStrategy.STANDARD,
    val additionalParams: Map<String, Any>? = null
)
```

---

### GenerationResult (@dataclass)

Returned by `ILanguageModel.generate()` and strategies.

| Field | Type | Meaning |
|-------|------|---------|
| `content` | str | The actual generated text |
| `tokens_used` | Optional[int] | Total tokens (prompt + response) |
| `cost` | Optional[float] | Estimated cost in USD |
| `metadata` | Optional[Dict] | Extra info (prompt_tokens, response_tokens, model name, etc.) |
| `strategy_used` | Optional[GenerationStrategy] | Which strategy was used |

**Kotlin equivalent:**
```kotlin
data class GenerationResult(
    val content: String,
    val tokensUsed: Int? = null,
    val cost: Float? = null,
    val metadata: Map<String, Any>? = null,
    val strategyUsed: GenerationStrategy? = null
)
```

---

### ITokenManager & Sub-Protocols

**ITokenCounter:**
```python
def count_tokens(text: str, model_name: str) -> int: ...
```
— Estimate token count for a string & model

**ICostEstimator:**
```python
def estimate_cost(tokens: int, model_name: str, is_output: bool = False) -> float: ...
```
— Convert token count to USD cost (input vs. output rates differ)

**IUsageTracker:**
```python
def log_usage(model_name: str, tokens: int, operation_type: str, is_output: bool = False) -> None: ...
def get_usage_summary() -> Dict[str, Any]: ...
```
— Record and retrieve session usage stats

**Combined ITokenManager:**
```python
class ITokenManager(ITokenCounter, ICostEstimator, IUsageTracker, Protocol): pass
```
— Implements all three—used by concrete `TokenManager` in token_utils.py

**Kotlin equivalent:**
```kotlin
interface ITokenCounter {
    fun countTokens(text: String, modelName: String): Int
}
interface ICostEstimator {
    fun estimateCost(tokens: Int, modelName: String, isOutput: Boolean = false): Float
}
interface IUsageTracker {
    fun logUsage(modelName: String, tokens: Int, operationType: String, isOutput: Boolean = false)
    fun getUsageSummary(): Map<String, Any>
}
interface ITokenManager : ITokenCounter, ICostEstimator, IUsageTracker
```

---

### IUserInteraction & Sub-Protocols

**IUserPrompt:**
```python
def prompt_continue() -> bool: ...  # Return True to continue, False to skip
def prompt_choice(message: str, choices: list[str]) -> str: ...  # Return selected choice
```

**IUserDisplay:**
```python
def display_info(message: str) -> None: ...
def display_error(error: str) -> None: ...
def display_warning(message: str) -> None: ...
```

**IUserInteraction:**
```python
class IUserInteraction(IUserPrompt, IUserDisplay, Protocol): pass
```

Implemented by `ConsoleUserInteraction` in services.py.

**Kotlin equivalent:**
```kotlin
interface IUserPrompt {
    fun promptContinue(): Boolean
    fun promptChoice(message: String, choices: List<String>): String
}
interface IUserDisplay {
    fun displayInfo(message: String)
    fun displayError(error: String)
    fun displayWarning(message: String)
}
interface IUserInteraction : IUserPrompt, IUserDisplay
```

---

### ILanguageModel (ABC)

**The only ABC in this file.** All model implementations must inherit from it.

```python
class ILanguageModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._validate_config()  # Called in parent __init__

    @abstractmethod
    def _validate_config(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult: ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]: ...

    @property
    @abstractmethod
    def provider(self) -> str: ...
```

**Key point:** When subclassing, call `super().__init__(config)` *after* setting instance fields (like `self.token_manager`), because the parent's `__init__` calls `self._validate_config()` which may need those fields.

**Kotlin equivalent:**
```kotlin
abstract class ILanguageModel(val config: ModelConfig) {
    init {
        validateConfig()
    }

    abstract fun validateConfig()
    abstract fun generate(prompt: String, vararg kwargs: Pair<String, Any>): GenerationResult
    abstract fun getModelInfo(): Map<String, Any>
    abstract val provider: String
}
```

---

### Other Protocols

**IModelValidator:**
```python
def validate_config(config: ModelConfig) -> None: ...
```
— Check that config is valid (not used widely in current codebase, but available)

**IApiKeyValidator:**
```python
def validate_key(api_key: str, provider: str) -> bool: ...
```
— Implemented by `ApiKeyValidator` in services.py; validates API keys before creating models

**IGenerationStrategy:**
```python
def generate(model: Any, prompt: str, config: ModelConfig) -> GenerationResult: ...
def supports_model(provider: str) -> bool: ...
```
— Implemented by `StandardGenerationStrategy` and `StreamingGenerationStrategy`

**IModelFactory:**
```python
def create_model(provider: str, config: ModelConfig) -> ILanguageModel: ...
def get_available_providers() -> list[str]: ...
def register_model(provider: str, model_class: type) -> None: ...
```
— Implemented by `ModelFactory` in models/model_factory.py

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `Enum` with `STANDARD = "standard"` | `enum class GenerationStrategy { STANDARD, STREAMING }` | GenerationStrategy |
| `@dataclass` on class | `data class ModelConfig(...)` | ModelConfig, GenerationResult |
| `Optional[str]` | `String?` | All type hints |
| `list[str]` | `List<String>` | IUserPrompt.prompt_choice() |
| `Dict[str, Any]` | `Map<String, Any>` | ITokenManager, return types |
| `Protocol` from typing | Kotlin `interface` (structural, no explicit impl) | All token/user/strategy protocols |
| `ABC` from abc | `abstract class` | ILanguageModel |
| `@abstractmethod` | `abstract fun` | ILanguageModel methods |
| `@property` + `@abstractmethod` | `abstract val` (computed) | ILanguageModel.provider |
| `def func(x: str) -> bool:` type hints | `fun func(x: String): Boolean` | All methods |

---

## How It Connects to Other Files

**Imports from:** Standard library only (`abc`, `enum`, `typing`, `dataclasses`)

**Imported by:** Nearly every file in core/ — this is the contract layer

**Flow:**
1. Services (services.py) implement token, user, validation Protocols
2. Models (models/claude_model.py) inherit from ILanguageModel
3. Strategies implement IGenerationStrategy
4. ModelFactory (model_factory.py) expects ILanguageModel, ITokenManager, IUserInteraction
5. DI container (dependency_injection.py) registers these and wires them together
6. main.py never touches interfaces directly—talks to DI container instead

**Key insight:** If you're adding a new interface (e.g., `INewFeature`), add it here, then implement it in services.py or models/, then register it in dependency_injection.py.

---

**[← Previous](index.md) | [Back to Index](index.md) | [Next →](exceptions.md)**

*Read next: exceptions.md — understand the error hierarchy*
