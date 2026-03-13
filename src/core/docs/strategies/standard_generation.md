# strategies/standard_generation.md

**📍 Reading Order:** #9 of 10 core docs | [← Back to Index](../index.md) | [Next: streaming_generation.md →](streaming_generation.md)

> Synchronous text generation strategy. Waits for full response before returning. Simple, reliable, predictable.

## What This File Does

The **Strategy Pattern** lets you swap generation approaches without changing model code:

- **StandardGenerationStrategy** — wait for complete response (synchronous)
- **StreamingGenerationStrategy** — stream chunks in real-time (asynchronous)

Both implement `IGenerationStrategy` interface, so `ClaudeModel` can pick either without caring which one. This file provides the standard (non-streaming) implementation.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `StandardGenerationStrategy` | class | Synchronous generation strategy |
| `.__init__(token_manager, user_interaction, logger)` | method | Initialize with DI-injected services |
| `.generate(model, prompt, config)` | method | Generate full response synchronously |
| `.supports_model(provider)` | method | Check if this strategy works with provider |

---

## StandardGenerationStrategy Class (Detailed)

### __init__

```python
def __init__(self, token_manager, user_interaction, logger):
    """Initialize the strategy with dependencies."""
    self.token_manager = token_manager
    self.user_interaction = user_interaction
    self.logger = logger
```

Receives DI-injected services. Doesn't create them.

### generate()

```python
def generate(self, model: Any, prompt: str, config: ModelConfig) -> GenerationResult:
    """
    Generate text using standard synchronous approach.

    Args:
        model: The language model instance (LangChain ChatAnthropic)
        prompt: Input text prompt
        config: Model configuration

    Returns:
        GenerationResult with content and metadata

    Raises:
        GenerationError: If generation fails
    """
    try:
        # 1. Prepare prompt with system message
        system_msg = config.system_message or "You are a helpful assistant."
        full_prompt = f"{system_msg}\n{prompt}"

        # 2. Token analysis BEFORE generation
        prompt_tokens = self.token_manager.count_tokens(full_prompt, config.model_name)
        est_cost = self.token_manager.estimate_cost(
            prompt_tokens,
            config.model_name,
            is_output=False
        )

        self.user_interaction.display_info("Prompt Analysis:")
        self.user_interaction.display_info(f"- Input tokens: {prompt_tokens}")
        self.user_interaction.display_info(f"- Estimated input cost: ${est_cost:.6f}")

        # 3. Create LangChain prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{input}")
        ])

        # 4. Format and invoke model (SYNCHRONOUS - waits for full response)
        formatted_prompt = prompt_template.format_messages(input=prompt)
        response = model.invoke(formatted_prompt)  # Blocks until done
        response_text = response.content

        # 5. Token analysis AFTER generation
        response_tokens = self.token_manager.count_tokens(response_text, config.model_name)
        response_cost = self.token_manager.estimate_cost(
            response_tokens,
            config.model_name,
            is_output=True
        )

        self.user_interaction.display_info("Response Analysis:")
        self.user_interaction.display_info(f"- Output tokens: {response_tokens}")
        self.user_interaction.display_info(f"- Estimated output cost: ${response_cost:.6f}")

        # 6. Log usage
        self.token_manager.log_usage(
            config.model_name,
            prompt_tokens,
            "standard_prompt",
            is_output=False
        )
        self.token_manager.log_usage(
            config.model_name,
            response_tokens,
            "standard_response",
            is_output=True
        )

        # 7. Show session summary
        summary = self.token_manager.get_usage_summary()
        self.user_interaction.display_info("Session Summary:")
        self.user_interaction.display_info(f"- Total tokens used: {summary['total_tokens']}")
        self.user_interaction.display_info(f"- Total estimated cost: ${summary['total_cost']:.6f}")

        # 8. Return result
        return GenerationResult(
            content=response_text,
            tokens_used=prompt_tokens + response_tokens,
            cost=est_cost + response_cost,
            strategy_used=GenerationStrategy.STANDARD,
            metadata={
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "model": config.model_name,
                "system_message": system_msg
            }
        )

    except Exception as e:
        error_msg = f"Standard generation failed: {e}"
        self.logger.error(error_msg)
        raise GenerationError(error_msg)
```

**Flow:**
1. Prepare system message + full prompt
2. Count input tokens, estimate input cost
3. Create LangChain prompt template with system message
4. **Call model.invoke() — SYNCHRONOUS, blocks until response arrives**
5. Count output tokens, estimate output cost
6. Log both input and output usage
7. Display session summary
8. Return GenerationResult with all metadata

**Key difference from streaming:** `model.invoke()` waits for the full response before returning. The user sees nothing until the response is complete.

### supports_model()

```python
def supports_model(self, provider: str) -> bool:
    """Check if this strategy supports the given provider."""
    # Standard generation works with all providers
    return True
```

Standard generation works with any model (no special requirements).

---

## LangChain ChatPromptTemplate Pattern

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    ("user", "{input}")
])

formatted_prompt = prompt_template.format_messages(input=prompt)
response = model.invoke(formatted_prompt)
```

**What's happening:**
1. `ChatPromptTemplate` is a LangChain abstraction for structuring prompts
2. Messages have roles: "system" (instructions to model), "user" (user's actual prompt)
3. `{input}` is a placeholder for the user's prompt
4. `.format_messages(input=prompt)` fills in the placeholder
5. `model.invoke()` sends the formatted messages to Claude

**Kotlin equivalent (conceptual):**
```kotlin
val promptTemplate = ChatPromptTemplate.fromMessages(
    listOf(
        ("system" to systemMsg),
        ("user" to "{input}")
    )
)
val formattedPrompt = promptTemplate.formatMessages(mapOf("input" to prompt))
val response = model.invoke(formattedPrompt)
```

---

## Token Counting & Cost Estimation

The strategy tracks token usage at multiple points:

**Before generation:**
```python
prompt_tokens = self.token_manager.count_tokens(full_prompt, model_name)
est_cost = self.token_manager.estimate_cost(prompt_tokens, model_name, is_output=False)
# Input tokens cost less than output tokens
```

**After generation:**
```python
response_tokens = self.token_manager.count_tokens(response_text, model_name)
response_cost = self.token_manager.estimate_cost(response_tokens, model_name, is_output=True)
# Output tokens cost more (5-300x depending on model)
```

**Logging:**
```python
self.token_manager.log_usage(model_name, prompt_tokens, "standard_prompt", is_output=False)
self.token_manager.log_usage(model_name, response_tokens, "standard_response", is_output=True)
```

Each operation type ("standard_prompt", "standard_response") is logged separately, allowing you to see which operations consume the most tokens.

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `def generate(self, model: Any, ...)` | `fun generate(model: Any, ...): GenerationResult` | Main method |
| `config.system_message or "default"` | `config.systemMessage ?: "default"` | Default value |
| `f"{msg} {var}"` | `"$msg $var"` string template | F-string |
| `model.invoke(prompt)` | `model.invoke(prompt)` or LangChain interop | Synchronous call |
| `response.content` | Property access | Getting response text |
| `try: ... except Exception as e:` | `try { ... } catch (e: Exception)` | Error handling |
| `self.logger.error()` | `Log.e(tag, msg)` | Logging |
| `raise GenerationError()` | `throw GenerationError()` | Raising exceptions |
| `self.user_interaction.display_info()` | `userInteraction.displayInfo()` | Method call |
| `.get_usage_summary()` | `.getUsageSummary()` | Method call |
| `f"${value:.6f}"` | `String.format("%.6f", value)` | Number formatting |

---

## How It Connects to Other Files

**Imports from:**
- `interfaces.py` — IGenerationStrategy, ModelConfig, GenerationResult, GenerationStrategy
- `exceptions.py` — GenerationError
- External: `langchain_core.prompts.ChatPromptTemplate`

**Imported by:**
- `models/claude_model.py` — used via _STRATEGIES registry

**Flow:**
1. **Strategy selection:**
   - `ClaudeModel.generate()` checks `config.generation_strategy`
   - Looks up `StandardGenerationStrategy` in _STRATEGIES dict
   - Instantiates strategy with token_manager, user_interaction, logger

2. **Generation:**
   - `ClaudeModel.generate()` calls `strategy.generate(model, prompt, config)`
   - Strategy builds LangChain prompt template
   - Strategy calls `model.invoke()` (synchronous)
   - LangChain's ChatAnthropic does the actual API call to Claude
   - Response returned to strategy

3. **Token tracking:**
   - Strategy calls `token_manager.count_tokens()`
   - Strategy calls `token_manager.estimate_cost()`
   - Strategy calls `token_manager.log_usage()`
   - User can later call `token_manager.get_usage_summary()` to see costs

4. **User display:**
   - Strategy calls `user_interaction.display_info()` for analysis
   - User sees token counts, estimated costs before, during, after

---

## Strengths & Weaknesses

**Strengths:**
- Simple: wait for full response, no complexity
- Predictable: response is complete before returning
- Full token counts: know exact tokens before and after
- Works everywhere: no special model requirements

**Weaknesses:**
- Not interactive: user sees nothing until response complete
- Slower perceived latency: complete response can take seconds
- No cancellation: can't stop mid-generation
- Large responses block: ties up thread waiting

**When to use:**
- Short responses (one-shot questions)
- Batch operations (don't need real-time feedback)
- When you don't have streaming support (older models)

**When NOT to use:**
- Long code generation (user wants to see early progress)
- Chat interface (user expects instant feedback)
- Real-time interaction (need to show chunks as they arrive)

---

**[← Previous](../models/claude_model.md) | [Back to Index](../index.md) | [Next →](streaming_generation.md)**

*Read next: streaming_generation.md — understand streaming generation (final core doc)*

---

## Relationship to StreamingGenerationStrategy

Both implement `IGenerationStrategy`:

| Aspect | Standard | Streaming |
|--------|----------|-----------|
| `model.invoke()` vs `model.stream()` | `.invoke()` (sync) | `.stream()` (generator) |
| When returns | After full response | After each chunk |
| User sees | Nothing until done | Real-time chunks |
| Use case | Batch, short responses | Chat, long responses |
| Latency feel | Slow (response time) | Fast (instant chunks) |

The **interface** is the same; the **implementation** differs in how it calls LangChain's model.
