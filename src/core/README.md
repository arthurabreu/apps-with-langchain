# Core Module - Refactored Architecture

This document describes the refactored core module that now follows SOLID principles and modern architectural patterns.

## 🏗️ Architecture Overview

The core module has been completely refactored to follow SOLID principles:

- **Single Responsibility Principle (SRP)**: Each class has a single, well-defined responsibility
- **Open/Closed Principle (OCP)**: Easy to extend with new models without modifying existing code
- **Liskov Substitution Principle (LSP)**: All models implement the same interface and can be substituted
- **Interface Segregation Principle (ISP)**: Interfaces are focused and specific
- **Dependency Inversion Principle (DIP)**: Classes depend on abstractions, not concrete implementations

## 📁 New Structure

```
src/core/
├── __init__.py                 # Clean API exports
├── interfaces.py               # Abstract interfaces and protocols
├── exceptions.py               # Custom exception hierarchy
├── services.py                 # Service implementations
├── dependency_injection.py     # DI container
├── token_utils.py             # Token management (existing)
├── utils.py                   # Backward compatibility utilities
├── models/
│   ├── __init__.py
│   ├── openai_model.py        # Refactored OpenAI model
│   ├── claude_model.py        # Refactored Claude model
│   └── model_factory.py       # Factory pattern implementation
└── README.md                  # This file
```

## 🔧 Key Components

### Interfaces (`interfaces.py`)
- `ILanguageModel`: Abstract base class for all language models
- `ITokenManager`: Protocol for token management operations
- `IUserInteraction`: Protocol for user interaction
- `ModelConfig`: Configuration data class
- `GenerationResult`: Result data class

### Services (`services.py`)
- `ConfigurationManager`: Centralized configuration management
- `ApiKeyValidator`: API key validation service
- `ConsoleUserInteraction`: Console-based user interaction
- `LoggingService`: Centralized logging setup

### Dependency Injection (`dependency_injection.py`)
- `DIContainer`: Simple dependency injection container
- Automatic service registration and resolution
- Singleton and factory patterns support

### Model Factory (`models/model_factory.py`)
- Factory pattern for creating model instances
- Automatic dependency injection
- Provider registration system

## 🚀 Usage Examples

### Basic Usage (Backward Compatible)
```python
from core import create_openai_model, create_claude_model

# Create models using convenience functions
openai_model = create_openai_model("gpt-4")
claude_model = create_claude_model("claude-3-sonnet-20240229")

# Generate text
result = openai_model.generate("Write a Python function")
print(result.content)
```

### Advanced Usage with DI Container
```python
from core import get_container, ModelConfig

# Get services from DI container
container = get_container()
factory = container.get_model_factory()

# Create custom configuration
config = ModelConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Create model through factory
model = factory.create_model("openai", config)
result = model.generate("Write a Python function")
```

### Adding New Model Providers
```python
from core import get_container
from core.interfaces import ILanguageModel

class MyCustomModel(ILanguageModel):
    # Implement required methods
    pass

# Register new provider
container = get_container()
factory = container.get_model_factory()
factory.register_model("custom", MyCustomModel)

# Use the new provider
model = factory.create_model("custom", config)
```

## 🔄 Migration Guide

### For Existing Code
The refactored module maintains backward compatibility through utility functions:

**Old way:**
```python
from core.openai_model import OpenAIModel
model = OpenAIModel("gpt-3.5-turbo", temperature=0.2)
```

**New way (backward compatible):**
```python
from core import create_openai_model
model = create_openai_model("gpt-3.5-turbo", temperature=0.2)
```

**Modern way:**
```python
from core import get_container, ModelConfig
container = get_container()
factory = container.get_model_factory()
config = ModelConfig(model_name="gpt-3.5-turbo", temperature=0.2)
model = factory.create_model("openai", config)
```

## 🧪 Testing

The new architecture makes testing much easier:

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

## 🎯 Benefits

1. **Maintainability**: Clear separation of concerns makes code easier to maintain
2. **Testability**: Dependency injection makes unit testing straightforward
3. **Extensibility**: Easy to add new model providers without changing existing code
4. **Consistency**: All models follow the same interface contract
5. **Error Handling**: Proper exception hierarchy for better error management
6. **Configuration**: Centralized configuration management
7. **Logging**: Structured logging throughout the application

## 🔍 Error Handling

The new architecture includes a comprehensive exception hierarchy:

- `LangChainAppError`: Base exception for all application errors
- `ModelConfigurationError`: Configuration-related errors
- `ApiKeyError`: API key validation errors
- `GenerationError`: Text generation errors
- `TokenManagementError`: Token management errors
- `UnsupportedProviderError`: Unknown provider errors

## 📊 Performance Considerations

- Services are created as singletons where appropriate
- Lazy initialization of expensive resources
- Proper resource cleanup and management
- Efficient dependency resolution

This refactored architecture provides a solid foundation for future development while maintaining compatibility with existing code.