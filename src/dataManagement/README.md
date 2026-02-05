# Data Management Package 📊

> **For Android Engineers**: Think of this as your **data layer** - like Room database, repositories, and data models in Android. This package handles all the "behind-the-scenes" data operations that keep track of costs, tokens, and model comparisons.

## What This Package Does

This package is responsible for managing all the data-related operations in our LangChain application. Just like how in Android you have a data layer that handles database operations, API responses, and data transformations, this package handles:

- **Token counting and cost tracking** (like tracking API usage in a mobile app)
- **Model comparison utilities** (like comparing different API endpoints)
- **Data persistence and retrieval** (like saving user preferences)

## Files in This Package

### 1. `token_utils.py` - The Token Accountant 💰

**What it does**: This is like having an accountant that tracks every API call you make and tells you how much it costs.

**Key Components**:

#### `TokenUsage` Class (Data Model)
```python
@dataclass
class TokenUsage:
    timestamp: str          # When the API call happened
    model: str             # Which AI model was used
    tokens_used: int       # How many tokens were consumed
    estimated_cost: float  # How much it cost in dollars
    operation_type: str    # What type of operation (chat, completion, etc.)
```

**Android Equivalent**: This is like a `@Entity` class in Room database - it defines the structure of data we want to store.

#### `TokenManager` Class (The Main Controller)

**Line-by-line breakdown of key methods**:

**Constructor and Pricing Data**:
```python
COST_PER_1M_TOKENS = {
    "gpt-5.2": {"input": 1.75, "output": 14.00, "description": "Advanced reasoning"},
    # ... more models
}
```
- **What this does**: Stores the current pricing for different AI models (as of 2026)
- **Why it matters**: Just like how you need to know data usage costs in mobile apps, we need to track AI usage costs
- **Android equivalent**: Like storing API rate limits or subscription pricing in constants

**Token Counting Method**:
```python
def count_tokens(self, text: str, model_name: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
```
- **Line 1**: Try to get the specific tokenizer for the model
- **Line 2**: Encode the text and count tokens
- **Line 4-5**: If model not found, use a default tokenizer
- **Android equivalent**: Like calculating the size of data before sending it over network

**Cost Calculation Method**:
```python
def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
    if model_name not in self.COST_PER_1M_TOKENS:
        return 0.0  # Unknown model, can't calculate cost
    
    pricing = self.COST_PER_1M_TOKENS[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```
- **Line 1-2**: Check if we know the pricing for this model
- **Line 5**: Calculate input cost (tokens ÷ 1 million × price per million)
- **Line 6**: Calculate output cost the same way
- **Line 7**: Return total cost
- **Android equivalent**: Like calculating data usage costs based on bytes transferred

**Usage Tracking Method**:
```python
def track_usage(self, model: str, tokens_used: int, operation_type: str = "chat") -> TokenUsage:
    cost = self.calculate_cost(tokens_used, 0, model)  # Simplified for input only
    usage = TokenUsage(
        timestamp=datetime.now().isoformat(),
        model=model,
        tokens_used=tokens_used,
        estimated_cost=cost,
        operation_type=operation_type
    )
    self.usage_history.append(usage)
    return usage
```
- **Line 1**: Calculate the cost for this usage
- **Lines 2-8**: Create a new TokenUsage record with current timestamp
- **Line 9**: Add to our history list
- **Line 10**: Return the usage record
- **Android equivalent**: Like logging analytics events or saving usage data to local database

### 2. `model_comparison.py` - The Model Benchmarker 📈

**What it does**: This helps you compare different AI models side by side, like comparing different API endpoints for performance and cost.

**Key Features**:
- Compare response quality between models
- Track performance metrics (speed, cost, accuracy)
- Generate comparison reports

**Android Equivalent**: Like having A/B testing utilities or performance monitoring tools that help you decide which API endpoint or library performs better.

## How to Use This Package

### Basic Token Tracking
```python
from dataManagement import TokenManager

# Create a token manager
token_manager = TokenManager()

# Count tokens in your text
text = "Hello, how are you today?"
token_count = token_manager.count_tokens(text, "gpt-4")
print(f"This text uses {token_count} tokens")

# Track usage and get cost
usage = token_manager.track_usage("gpt-4", token_count, "chat")
print(f"This will cost approximately ${usage.estimated_cost:.4f}")
```

### Getting Usage Statistics
```python
# Get total usage for today
total_cost = token_manager.get_daily_cost()
print(f"Today's total cost: ${total_cost:.2f}")

# Get usage by model
usage_by_model = token_manager.get_usage_by_model()
for model, stats in usage_by_model.items():
    print(f"{model}: {stats['total_tokens']} tokens, ${stats['total_cost']:.2f}")
```

## Why This Matters for Android Developers

1. **Cost Control**: Just like monitoring data usage in mobile apps, this helps you monitor AI API costs
2. **Performance Tracking**: Similar to APM tools, this tracks which models perform best
3. **Data Persistence**: Like Room database, this saves usage history for analysis
4. **Resource Management**: Like memory management in Android, this helps manage token limits

## Common Patterns You'll Recognize

- **Data Classes**: `TokenUsage` is like Android's data classes or entities
- **Repository Pattern**: `TokenManager` acts like a repository that manages data operations
- **Observer Pattern**: Usage tracking is like analytics event tracking
- **Strategy Pattern**: Different cost calculation strategies for different models

## Files Structure
```
dataManagement/
├── __init__.py          # Package exports (like module.gradle)
├── token_utils.py       # Main token management logic
├── model_comparison.py  # Model comparison utilities
└── README.md           # This documentation
```

This package is the foundation that keeps track of all your AI interactions, just like how your Android app's data layer keeps track of user interactions and app state!