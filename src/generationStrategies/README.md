# Generation Strategies Package ⚡

> **For Android Engineers**: Think of this as your **strategy pattern implementations** - like different algorithms for image processing, network requests, or data parsing in Android. This package handles different approaches to generating AI responses.

## What This Package Does

This package is responsible for managing different text generation strategies in our LangChain application. Just like how in Android you might have different strategies for loading images (Glide vs Picasso) or network requests (Retrofit vs Volley), this package handles:

- **Standard Generation Strategy** (synchronous, wait for complete response)
- **Streaming Generation Strategy** (real-time, word-by-word output)
- **Strategy Selection Logic** (choosing the right approach for each situation)
- **Token Tracking and Cost Management** (monitoring usage across all strategies)

## Files in This Package

### 1. `strategies/standard_generation.py` - The Standard Approach 📝

**What it does**: This is like making a regular API call and waiting for the complete response before showing it to the user.

**Key Components**:

#### `StandardGenerationStrategy` Class - Line by Line

**Constructor**:
```python
def __init__(self, token_manager, user_interaction, logger):
    self.token_manager = token_manager      # Tracks token usage and costs
    self.user_interaction = user_interaction # Handles user prompts and display
    self.logger = logger                    # Logs operations and errors
```

**Android Equivalent**: Like injecting dependencies in your Activity or Fragment constructor - you get the services you need to do your work.

**Main Generation Method**:
```python
def generate(self, model: Any, prompt: str, config: ModelConfig) -> GenerationResult:
    try:
        # Step 1: Prepare the prompt with system message
        system_msg = config.system_message or "You are a helpful assistant."
        full_prompt = f"{system_msg}\n{prompt}"
        
        # Step 2: Analyze tokens BEFORE making the API call
        prompt_tokens = self.token_manager.count_tokens(full_prompt, config.model_name)
        est_cost = self.token_manager.estimate_cost(prompt_tokens, config.model_name, is_output=False)
        
        # Step 3: Show user what this will cost
        self.user_interaction.display_info("Prompt Analysis:")
        self.user_interaction.display_info(f"- Input tokens: {prompt_tokens}")
        self.user_interaction.display_info(f"- Estimated input cost: ${est_cost:.6f}")
```

- **Lines 3-4**: Combine system instructions with user prompt (like preparing request headers + body)
- **Lines 6-7**: Count tokens and estimate cost BEFORE making the expensive API call
- **Lines 9-12**: Show user the cost analysis (like showing data usage before downloading)
- **Android equivalent**: Like checking network conditions and data usage before making an expensive API call

**API Call and Response Processing**:
```python
        # Step 4: Create structured prompt and make API call
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{input}")
        ])
        
        formatted_prompt = prompt_template.format_messages(input=prompt)
        response = model.invoke(formatted_prompt)  # This is the actual API call
        response_text = response.content
        
        # Step 5: Analyze the response tokens and cost
        response_tokens = self.token_manager.count_tokens(response_text, config.model_name)
        response_cost = self.token_manager.estimate_cost(response_tokens, config.model_name, is_output=True)
```

- **Lines 2-5**: Structure the prompt using LangChain's template system
- **Line 7**: Make the actual API call (this is where the magic happens)
- **Lines 10-11**: Count tokens in the response and calculate actual cost
- **Android equivalent**: Like making a Retrofit API call and then processing the response

**Usage Tracking and Results**:
```python
        # Step 6: Log usage for analytics and cost tracking
        self.token_manager.log_usage(config.model_name, prompt_tokens, "standard_prompt", is_output=False)
        self.token_manager.log_usage(config.model_name, response_tokens, "standard_response", is_output=True)
        
        # Step 7: Show session summary
        summary = self.token_manager.get_usage_summary()
        self.user_interaction.display_info("Session Summary:")
        self.user_interaction.display_info(f"- Total tokens used: {summary['total_tokens']}")
        self.user_interaction.display_info(f"- Total estimated cost: ${summary['total_cost']:.6f}")
        
        # Step 8: Return structured result
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
```

- **Lines 2-3**: Log usage for both input and output (like analytics tracking)
- **Lines 5-8**: Show user their total session usage (like showing total data used)
- **Lines 10-20**: Return a structured result with all the metadata
- **Android equivalent**: Like logging analytics events and returning a structured response object

### 2. `strategies/streaming_generation.py` - The Real-Time Approach 🌊

**What it does**: This is like streaming video or music - you see/hear content as it arrives, not after everything is downloaded.

**Key Components**:

#### `StreamingGenerationStrategy` Class

**The Streaming Magic**:
```python
def generate(self, model: Any, prompt: str, config: ModelConfig) -> GenerationResult:
    # ... (same setup as standard generation)
    
    # The key difference: Stream the response
    response_chunks = []
    print("\n[STREAMING] ", end="", flush=True)
    
    try:
        # Try streaming if the model supports it
        for chunk in model.stream(formatted_prompt):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)  # Show immediately
                response_chunks.append(chunk.content)     # Save for later
    except AttributeError:
        # Fallback: if streaming not supported, use regular generation
        self.user_interaction.display_warning("Streaming not supported, falling back to standard generation")
        response = model.invoke(formatted_prompt)
        response_text = response.content
        print(response_text)
        response_chunks = [response_text]
    
    print("\n")  # New line after streaming complete
    
    # Combine all chunks into final response
    response_text = "".join(response_chunks)
```

**Line-by-line breakdown**:
- **Line 6**: Start streaming indicator (like showing a progress bar)
- **Lines 9-12**: For each chunk that arrives, immediately display it AND save it
- **Lines 13-18**: Graceful fallback if streaming isn't supported
- **Line 22**: Combine all chunks into the final complete response
- **Android equivalent**: Like streaming video where you show frames as they arrive, but also buffer them for replay

**Provider Support Check**:
```python
def supports_model(self, provider: str) -> bool:
    """Check if this strategy supports the given provider."""
    # Most modern providers support streaming
    supported_providers = ["openai", "anthropic"]
    return provider.lower() in supported_providers
```

**Android Equivalent**: Like checking if a device supports certain features (camera, GPS, etc.) before trying to use them.

## How to Use This Package

### Using Standard Generation
```python
from generationStrategies import StandardGenerationStrategy
from systemCore import ModelConfig, GenerationStrategy

# Create strategy instance
strategy = StandardGenerationStrategy(token_manager, user_interaction, logger)

# Configure model
config = ModelConfig(
    model_name="gpt-4",
    temperature=0.7,
    generation_strategy=GenerationStrategy.STANDARD
)

# Generate response
result = strategy.generate(model, "Hello, how are you?", config)
print(f"Response: {result.content}")
print(f"Cost: ${result.cost:.6f}")
```

### Using Streaming Generation
```python
from generationStrategies import StreamingGenerationStrategy

# Create streaming strategy
streaming_strategy = StreamingGenerationStrategy(token_manager, user_interaction, logger)

# Same config, different strategy
config.generation_strategy = GenerationStrategy.STREAMING

# Generate with real-time output
result = streaming_strategy.generate(model, "Tell me a story", config)
# You'll see the story appear word by word as it's generated!
```

### Strategy Selection Logic
```python
def choose_strategy(user_preference, model_provider):
    """Choose the right strategy based on context."""
    
    if user_preference == "fast_feedback":
        if StreamingGenerationStrategy().supports_model(model_provider):
            return StreamingGenerationStrategy(token_manager, user_interaction, logger)
    
    # Default to standard for reliability
    return StandardGenerationStrategy(token_manager, user_interaction, logger)

# Usage
strategy = choose_strategy("fast_feedback", "openai")
result = strategy.generate(model, prompt, config)
```

## Why This Matters for Android Developers

1. **Strategy Pattern**: Just like choosing different image loading strategies (Glide vs Picasso)
2. **Real-time Updates**: Like streaming data in RecyclerView or live chat applications
3. **Graceful Degradation**: Like falling back to cached data when network fails
4. **Cost Monitoring**: Like tracking data usage or battery consumption
5. **User Experience**: Like showing progress indicators during long operations

## Common Patterns You'll Recognize

- **Strategy Pattern**: Different algorithms for the same task (generation)
- **Template Method**: Common setup/teardown with different core logic
- **Observer Pattern**: Real-time updates during streaming
- **Fallback Pattern**: Graceful degradation when features aren't supported
- **Analytics Pattern**: Tracking usage and performance metrics

## Performance Considerations

### Standard Generation
- **Pros**: Reliable, works with all models, complete response at once
- **Cons**: User waits for entire response, no feedback during generation
- **Best for**: Short responses, batch processing, when reliability is key

### Streaming Generation  
- **Pros**: Immediate feedback, better user experience, feels faster
- **Cons**: More complex, not all models support it, harder to handle errors
- **Best for**: Long responses, interactive chat, when user experience is priority

## Files Structure
```
generationStrategies/
├── __init__.py                    # Package exports
├── strategies/
│   ├── __init__.py               # Strategy exports  
│   ├── standard_generation.py   # Synchronous generation
│   └── streaming_generation.py  # Real-time streaming generation
└── README.md                     # This documentation
```

This package is like having different rendering engines in Android - you can choose the one that best fits your needs while maintaining the same interface for the rest of your app!