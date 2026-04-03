# strategies/streaming_generation.md

**📍 Reading Order:** #10 of 10 core docs | [← Back to Index](../index.md) | [Optional next: token_utils.md →](../token_utils.md)

> Streaming text generation strategy. Outputs chunks in real-time, showing response as it arrives. Feels fast and interactive.

## What This File Does

This strategy uses **generators** — Python's way of writing functions that yield multiple values over time. Instead of waiting for a complete response:

```python
# Standard (wait)
result = model.invoke(prompt)  # Blocks
response_text = result.content

# Streaming (real-time)
for chunk in model.stream(prompt):  # Yields chunks as they arrive
    print(chunk.content, end="", flush=True)
```

Streaming provides real-time feedback—user sees the response character by character as the model generates it.

---

## Classes & Functions

| Name | Type | What It Does |
|------|------|--------------|
| `StreamingGenerationStrategy` | class | Real-time streaming generation strategy |
| `.__init__(token_manager, user_interaction, logger)` | method | Initialize with DI-injected services |
| `.generate(model, prompt, config)` | method | Generate and stream response in real-time |
| `.supports_model(provider)` | method | Check if this strategy works with provider |

---

## StreamingGenerationStrategy Class (Detailed)

### __init__

```python
def __init__(self, token_manager, user_interaction, logger):
    """Initialize the strategy with dependencies."""
    self.token_manager = token_manager
    self.user_interaction = user_interaction
    self.logger = logger
```

Same as StandardGenerationStrategy. Receives DI-injected services.

### generate()

```python
def generate(self, model: Any, prompt: str, config: ModelConfig) -> GenerationResult:
    """
    Generate text using streaming approach.

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

        self.user_interaction.display_info("Streaming Generation:")
        self.user_interaction.display_info(f"- Input tokens: {prompt_tokens}")
        self.user_interaction.display_info(f"- Estimated input cost: ${est_cost:.6f}")
        self.user_interaction.display_info("- Starting stream...")

        # 3. Create LangChain prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{input}")
        ])

        formatted_prompt = prompt_template.format_messages(input=prompt)

        # 4. Stream the response (ASYNCHRONOUS - yields chunks)
        response_chunks = []
        print("\n[STREAMING] ", end="", flush=True)

        try:
            # Try streaming if supported (most modern models support it)
            for chunk in model.stream(formatted_prompt):  # Generator yields chunks
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)  # Print as it arrives
                    response_chunks.append(chunk.content)  # Collect for later analysis
        except AttributeError:
            # Fallback: if streaming not supported, use synchronous generation
            self.user_interaction.display_warning(
                "Streaming not supported, falling back to standard generation"
            )
            response = model.invoke(formatted_prompt)
            response_text = response.content
            print(response_text)
            response_chunks = [response_text]

        print("\n")  # New line after streaming completes

        # 5. Combine all chunks into single response
        response_text = "".join(response_chunks)

        # 6. Token analysis AFTER generation
        response_tokens = self.token_manager.count_tokens(response_text, config.model_name)
        response_cost = self.token_manager.estimate_cost(
            response_tokens,
            config.model_name,
            is_output=True
        )

        self.user_interaction.display_info("Stream Complete:")
        self.user_interaction.display_info(f"- Output tokens: {response_tokens}")
        self.user_interaction.display_info(f"- Estimated output cost: ${response_cost:.6f}")

        # 7. Log usage
        self.token_manager.log_usage(
            config.model_name,
            prompt_tokens,
            "streaming_prompt",
            is_output=False
        )
        self.token_manager.log_usage(
            config.model_name,
            response_tokens,
            "streaming_response",
            is_output=True
        )

        # 8. Return result
        return GenerationResult(
            content=response_text,
            tokens_used=prompt_tokens + response_tokens,
            cost=est_cost + response_cost,
            strategy_used=GenerationStrategy.STREAMING,
            metadata={
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "model": config.model_name,
                "system_message": system_msg,
                "chunks_count": len(response_chunks)
            }
        )

    except Exception as e:
        error_msg = f"Streaming generation failed: {e}"
        self.logger.error(error_msg)
        raise GenerationError(error_msg)
```

**Key differences from StandardGenerationStrategy:**

1. **Streaming loop:**
   ```python
   for chunk in model.stream(formatted_prompt):  # Generator
       if hasattr(chunk, 'content') and chunk.content:
           print(chunk.content, end="", flush=True)  # Print immediately
           response_chunks.append(chunk.content)  # Save for analysis
   ```
   Uses `model.stream()` instead of `model.invoke()`. Returns a **generator** that yields chunks.

2. **Immediate output:**
   ```python
   print(chunk.content, end="", flush=True)
   ```
   `end=""` prevents newline (stream on one line), `flush=True` forces output immediately (don't buffer).

3. **Fallback to standard:**
   ```python
   try:
       for chunk in model.stream(...):
           ...
   except AttributeError:
       # Model doesn't support streaming, use synchronous
       response = model.invoke(...)
   ```
   If model doesn't support streaming, gracefully fall back to `invoke()`.

4. **Chunk metadata:**
   ```python
   "chunks_count": len(response_chunks)
   ```
   Tracks how many chunks were received (useful for debugging/analysis).

### supports_model()

```python
def supports_model(self, provider: str) -> bool:
    """Check if this strategy supports the given provider."""
    supported_providers = ["anthropic"]
    return provider.lower() in supported_providers
```

Streaming only works with providers that support it (currently Anthropic Claude). Falls back to standard generation if provider doesn't support streaming.

---

## Streaming Patterns Explained

### 1. Generators with `yield`

Python generators are functions that `yield` values:

```python
def my_generator():
    yield "a"
    yield "b"
    yield "c"

for value in my_generator():  # Consumes values one at a time
    print(value)  # Prints: a b c
```

LangChain's `model.stream()` is a generator that yields chunks:

```python
for chunk in model.stream(prompt):
    process(chunk)
```

**Kotlin equivalent:**
```kotlin
fun myGenerator(): Sequence<String> {
    return sequenceOf("a", "b", "c")
}

for (value in myGenerator()) {
    println(value)
}
```

### 2. Real-Time Console Output

```python
print(chunk.content, end="", flush=True)
```

- `chunk.content` — the text chunk
- `end=""` — don't add newline after each chunk
- `flush=True` — write to console immediately (don't buffer)

Without `flush=True`, Python might buffer output, delaying what user sees by seconds.

**Kotlin equivalent:**
```kotlin
print(chunk.content)
System.out.flush()
```

### 3. Attribute Safety with `hasattr`

```python
if hasattr(chunk, 'content') and chunk.content:
    # Use chunk.content safely
```

**Why?** Different LangChain versions or models might return different chunk types. `hasattr()` checks if attribute exists before accessing it.

**Kotlin equivalent:**
```kotlin
if (chunk is ContentChunk && chunk.content != null) {
    // Use chunk.content
}
```

### 4. Fallback Pattern

```python
try:
    for chunk in model.stream(...):
        ...
except AttributeError:
    # Streaming not supported, use synchronous
    response = model.invoke(...)
```

If streaming raises `AttributeError` (attribute missing), fall back to synchronous. This makes streaming **optional**—if not available, code still works.

---

## Token Counting in Streaming

Streaming makes token counting tricky:

1. **Input tokens** — count prompt before streaming (accurate)
2. **Output tokens** — stream chunks in real-time, token count unknown until done
3. **Solution** — count full response after collecting all chunks

```python
# Before streaming
prompt_tokens = count_tokens(full_prompt)

# After streaming
response_text = "".join(response_chunks)  # Combine all chunks
response_tokens = count_tokens(response_text)  # Count combined result
```

Cost is estimated based on full collected response, not per-chunk.

---

## Python → Kotlin Cheat Sheet (This File)

| Python | Kotlin | Where in this file |
|--------|--------|------------------|
| `for chunk in generator:` | `for (chunk in generator)` | Streaming loop |
| `model.stream(prompt)` | `model.stream(prompt)` (LangChain interop) | Streaming call |
| `print(..., end="", flush=True)` | `print(...); System.out.flush()` | Real-time output |
| `hasattr(obj, 'attr')` | `obj is Type && obj.attr != null` | Attribute safety |
| `chunk.content` | Property access | Getting chunk text |
| `except AttributeError:` | `catch (e: AttributeError)` | Specific exception |
| `response_chunks = []` | `val responseChunks = mutableListOf<String>()` | List accumulation |
| `.append(item)` | `.add(item)` | List append |
| `"".join(list)` | `list.joinToString("")` | Join strings |
| `try: ... except Exception:` | `try { ... } catch (e: Exception)` | Error handling |
| `f"${value:.6f}"` | `String.format("%.6f", value)` | Number formatting |
| `metadata["chunks_count"]` | `metadata["chunksCount"]` | Dictionary access |

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
   - If `STREAMING`, looks up `StreamingGenerationStrategy` in _STRATEGIES
   - Instantiates strategy with token_manager, user_interaction, logger

2. **Streaming generation:**
   - `ClaudeModel.generate()` calls `strategy.generate(model, prompt, config)`
   - Strategy calls `model.stream()` (LangChain)
   - LangChain's ChatAnthropic streams chunks from Claude API
   - Strategy prints each chunk immediately to console
   - User sees response in real-time

3. **Response collection:**
   - Strategy collects all chunks in `response_chunks` list
   - After streaming completes, joins them into `response_text`
   - Counts tokens on full collected response

4. **Token tracking:**
   - Same as StandardGenerationStrategy
   - Logs input and output usage separately

---

## Strengths & Weaknesses

**Strengths:**
- Interactive: user sees response as it arrives
- Fast-feeling: chunks appear immediately (perceived responsiveness)
- Engaging: visual feedback of model working
- Great for long responses: don't wait for complete response

**Weaknesses:**
- Complex: need to handle chunks, generators, fallbacks
- Partial responses: might need to wait for stream to complete for accurate token count
- Cannotcancel easily: once streaming starts, hard to interrupt
- Model-dependent: not all models support streaming

**When to use:**
- Chat interfaces (user expects real-time)
- Long code generation (user wants to see progress)
- Educational demos (show real-time response)
- User-facing applications (better UX)

**When NOT to use:**
- Simple, short responses (standard is simpler)
- Models that don't support streaming (fallback to standard anyway)
- When you need final response before processing (stream waits anyway)

---

## Relationship to StandardGenerationStrategy

Both implement `IGenerationStrategy`:

| Aspect | Standard | Streaming |
|--------|----------|-----------|
| Call | `model.invoke()` | `model.stream()` |
| Return type | Single response | Generator of chunks |
| User sees | Nothing until done | Real-time chunks |
| Feels | Slow (response latency) | Fast (instant feedback) |
| Use case | Batch, QA | Chat, real-time |
| Complexity | Simple | More complex (generators) |

**Design principle:** Both strategies follow the same interface (`IGenerationStrategy`), so `ClaudeModel.generate()` doesn't care which one is used. Swapping strategies requires only changing config.generation_strategy, not any code.

---

## Example Output

```
[STREAMING] The answer is simple. Python generators are functions that yield
values lazily. Instead of creating an entire list in memory, generators produce
values one at a time. This makes them memory-efficient for large sequences...

Stream Complete:
- Output tokens: 45
- Estimated output cost: $0.000675
```

User sees the response building in real-time as the model generates it.

---

**[← Previous](standard_generation.md) | [Back to Index](../index.md)**

---

## 🎉 You've Completed the Core Learning Path!

**Congratulations!** You've read all 10 core docs and understand:
- ✅ Contracts & interfaces
- ✅ Exception hierarchy
- ✅ Dependency injection
- ✅ Service implementations
- ✅ Model creation (factory pattern)
- ✅ Model implementation
- ✅ Generation strategies (sync & async)

**Next steps:**
- 📚 **Optional reference docs** (explore as needed):
  - [token_utils.md](../token_utils.md) — for token tracking
  - [langchain_huggingface_local.md](../langchain_huggingface_local.md) — for local models
  - [model_comparison.md](../model_comparison.md) — for benchmarking

- 💻 **Apply what you learned:**
  - Open `src/main.py` and trace how it uses the DI container
  - Look at the actual source files alongside these docs
  - Try modifying a strategy or adding a new service

- 🔗 **Go back to index:** [← Back to Index](../index.md)
