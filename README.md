# LangChain Personal Agent with Claude

A modern Python application for building LLM agents with LangChain and Claude (with experimental support for local HuggingFace models). This project demonstrates clean architecture principles, dependency injection, and type-safe API contracts.

## 📱 Quick Start for Mobile Developers (Android/Kotlin)

If you're coming from an Android background, here's how to think about this project:

1. **Dependency Injection**: We use a custom DI Container in `src/core/di/`. It's like a simplified version of **Koin** or **Hilt**.
2. **Activity/Orchestration**: `src/main.py` is your `Launcher`. It hands off control to `InteractiveCLI` in `src/core/cli_service.py`, which acts as your `MainActivity` logic.
3. **Data Classes**: We use Python `@dataclass` for model configurations (`ModelConfig`). This is equivalent to Kotlin `data class`.
4. **Interfaces**: Check `src/core/interfaces/`. We use `Protocols` and `ABCs`, which are the Python equivalent of Kotlin `interface` and `abstract class`.
5. **Environment**: The `.env` file is like your `local.properties` or `BuildConfig` - keep your API keys there!

For a deeper dive, see our [Developer Guide for Android Devs](src/docs/DEVELOPER_GUIDE.md).

## 🚀 Initial Setup

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. **Create/update the `.env` file** in the project root:

```env
# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Hugging Face API Key (for local HuggingFace models)
HUGGINGFACE_API_KEY=your-huggingface-key-here
```

### 3. Run the Application

```bash
python src/main.py
```

## 📁 Project Structure

```
├── src/                                  # All application code
│   ├── main.py                          # Entry point / bootstrapper
│   ├── data/                            # Persistent data (costs, token usage)
│   ├── logs/                            # Application logs
│   ├── responses/                       # Generated outputs and exports
│   ├── samples/                         # Sample task files
│   ├── scripts/                         # Utility scripts (cost tracking hooks)
│   ├── docs/                            # Project documentation
│   ├── android_agent/                   # Android code generation agent
│   ├── prompts/                         # Prompt templates and contexts
│   └── core/                            # Core business logic
│       ├── __init__.py                  # Clean API exports
│       ├── cli_service.py               # Interactive CLI orchestrator
│       ├── config.py                    # Global constants and pricing
│       ├── services/                    # Service implementations (config, logging, etc.)
│       ├── interfaces/                  # Abstract contracts (protocols, ABCs)
│       ├── exceptions/                  # Custom exception hierarchy
│       ├── di/                          # Dependency injection container
│       ├── utils/                       # Utilities (token management, helpers)
│       ├── cost_tracker/                # Cost tracking service
│       ├── excel_manipulation/          # Excel export service
│       ├── hf_local/                    # Local HuggingFace model support
│       ├── hf_model_manager/            # HuggingFace model download/selection
│       ├── models/                      # LLM model implementations
│       │   ├── claude_model.py
│       │   ├── minimax_model.py
│       │   └── model_factory.py
│       └── strategies/                  # Generation strategies (standard, streaming)
├── .env                                 # Environment variables (NOT committed)
├── .gitignore                           # Configured to ignore .env and logs
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🏗️ Architecture

The application follows SOLID principles for maintainability and testability:

### SOLID Principles

- **Single Responsibility Principle (SRP)**: Each class has a single, well-defined responsibility
- **Open/Closed Principle (OCP)**: Easy to extend with new models without modifying existing code
- **Liskov Substitution Principle (LSP)**: All models implement the same interface and can be substituted
- **Interface Segregation Principle (ISP)**: Interfaces are focused and specific
- **Dependency Inversion Principle (DIP)**: Classes depend on abstractions, not concrete implementations

### Key Components

**Interfaces** (`src/core/interfaces/`)
- `ILanguageModel`: Abstract base for all language models
- `ITokenManager`: Protocol for token management operations
- `IUserInteraction`: Protocol for user interaction
- `ModelConfig`: Configuration data class
- `GenerationResult`: Result data class

**Services** (`src/core/services/`)
- `ConfigurationManager`: Centralized configuration management
- `ApiKeyValidator`: API key validation service
- `ConsoleUserInteraction`: Console-based user interaction
- `LoggingService`: Centralized logging setup

**Dependency Injection** (`src/core/di/`)
- `DIContainer`: Simple DI container
- Automatic service registration and resolution
- Singleton and factory patterns support

**Model Factory** (`src/core/models/model_factory.py`)
- Factory pattern for creating model instances
- Automatic dependency injection
- Provider registration system (extensible)

### Model Providers

- **`anthropic`** → `ClaudeModel` (via `langchain_anthropic.ChatAnthropic`)
- **`minimax`** / **`huggingface`** → `MiniMaxModel` (local HuggingFace transformers)

### Generation Strategies

- **`STANDARD`** → Synchronous generation via `model.invoke()`
- **`STREAMING`** → Streaming generation via `model.stream()`

## 💡 Usage Examples

### Basic Usage (Backward Compatible)

```python
from core import create_claude_model, prompt_continue

# Create a Claude model
model = create_claude_model("claude-3-haiku-20240307", temperature=0.2)

# Generate text
result = model.generate("Write a Python function to calculate fibonacci")
print(result.content)

# Continue generation
prompt_continue()
```

### Advanced Usage with DI Container

```python
from core import get_container, ModelConfig, GenerationStrategy

# Get services from DI container
container = get_container()
factory = container.get_model_factory()

# Create custom configuration
config = ModelConfig(
    model_name="claude-3-haiku-20240307",
    temperature=0.7,
    max_tokens=1024,
    generation_strategy=GenerationStrategy.STREAMING
)

# Create model through factory
model = factory.create_model("anthropic", config)
result = model.generate("Explain quantum computing")
print(result.content)
```

### Adding New Model Providers

```python
from core import get_container
from core.interfaces import ILanguageModel, ModelConfig, GenerationResult

class MyCustomModel(ILanguageModel):
    def _validate_config(self, config: ModelConfig) -> None:
        pass  # Custom validation

    def generate(self, prompt: str) -> GenerationResult:
        # Custom implementation
        pass

    def get_model_info(self) -> dict:
        return {"provider": "custom", "model": "my-model"}

    @property
    def provider(self) -> str:
        return "custom"

# Register new provider
container = get_container()
factory = container.get_model_factory()
factory.register_model("custom", MyCustomModel)

# Use the new provider
config = ModelConfig(model_name="my-model")
model = factory.create_model("custom", config)
```

## 🧪 Testing

The architecture makes testing straightforward:

```python
from core import reset_container, get_container
from core.interfaces import IUserInteraction

class MockUserInteraction:
    def prompt_continue(self) -> bool:
        return True  # Always continue in tests

# Setup test container
reset_container()
container = get_container()
container.register_instance(IUserInteraction, MockUserInteraction())

# Now all models will use the mock interaction
```

## 🔒 Security

- ✅ The `.env` file is in `.gitignore` - never committed to version control
- ✅ Use different `.env` files for different environments
- ✅ Never hardcode API keys in the code
- ✅ Use safe defaults when variables are not set

## 🎯 Benefits

1. **Maintainability**: Clear separation of concerns makes code easier to maintain
2. **Testability**: Dependency injection makes unit testing straightforward
3. **Extensibility**: Easy to add new model providers without changing existing code
4. **Consistency**: All models follow the same interface contract
5. **Error Handling**: Proper exception hierarchy for better error management
6. **Configuration**: Centralized configuration management
7. **Logging**: Structured logging throughout the application

## 🔍 Error Handling

The application includes a comprehensive exception hierarchy:

- `LangChainAppError`: Base exception for all application errors
- `ConfigurationError`: Configuration-related errors (subclasses: `ModelConfigurationError`, `ApiKeyError`, etc.)
- `ModelError`: Model-related errors (subclasses: `ModelInitializationError`, `UnsupportedProviderError`, etc.)
- `GenerationError`: Text generation errors (subclasses: `PromptError`, `StreamingError`, `TokenLimitExceededError`, etc.)
- `TokenManagementError`: Token management errors (subclasses: `TokenCountingError`, `CostEstimationError`, etc.)
- `ServiceError`: Service-related errors (subclasses: `DependencyInjectionError`, `ServiceNotFoundError`, etc.)

## 🆘 Troubleshooting

### Error: "API key not configured"
- Verify the `.env` file exists in the project root
- Confirm API keys are set correctly in `.env`
- Run `python src/main.py` to check the status of your API keys

### Error: "Module not found"
- Execute `pip install -r requirements.txt`
- Verify you're in the correct virtual environment
- Check that all source files are present in `src/core/`

### Error: "No such file or directory: src/data/..."
- Verify that `src/data/`, `src/logs/`, and `src/responses/` directories exist
- The application should create them automatically on first run

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Anthropic Claude API Docs](https://docs.anthropic.com/)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [python-dotenv Documentation](https://python-dotenv.readthedocs.io/)

---

**Created with ❤️ as a personal agent for exploring LLM capabilities with clean architecture principles.**
