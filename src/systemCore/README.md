# System Core Package 🏗️

> **For Android Engineers**: Think of this as your **core framework layer** - like Android's base classes, interfaces, dependency injection (Dagger/Hilt), and system services. This package provides the foundation that everything else builds upon.

## What This Package Does

This package is responsible for providing the core system architecture and foundation classes for our LangChain application. Just like how Android has core framework classes (Activity, Fragment, Context) and dependency injection systems, this package handles:

- **Abstract Interfaces and Contracts** (like Android's interfaces and abstract classes)
- **Exception Definitions** (like Android's custom exception hierarchy)
- **Dependency Injection Container** (like Dagger/Hilt modules and components)
- **Core Services and Utilities** (like Android's system services)
- **Configuration Management** (like SharedPreferences or configuration providers)

## Files in This Package

### 1. `interfaces.py` - The Contract Definitions 📋

**What it does**: This defines all the interfaces and data structures that other parts of the app must follow - like defining API contracts.

**Key Components**:

#### `GenerationStrategy` Enum
```python
class GenerationStrategy(Enum):
    """Strategies for text generation."""
    STANDARD = "standard"
    STREAMING = "streaming"
    BATCH = "batch"
```

**Android Equivalent**: Like defining enum constants for different states or modes in your app.

#### `ModelConfig` Data Class
```python
@dataclass
class ModelConfig:
    """Configuration for language models."""
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 512
    api_key: Optional[str] = None
    system_message: Optional[str] = None
    generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD
    additional_params: Optional[Dict[str, Any]] = None
```

**Android Equivalent**: Like a data class or Parcelable that holds configuration data - similar to how you might create a `NetworkConfig` or `UserPreferences` class.

#### `GenerationResult` Data Class
```python
@dataclass
class GenerationResult:
    """Result of a text generation operation."""
    content: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    strategy_used: Optional[GenerationStrategy] = None
```

**Android Equivalent**: Like a response object from an API call - contains the main data plus metadata about the operation.

#### Core Interfaces

**`ILanguageModel` Interface**:
```python
class ILanguageModel(Protocol):
    """Interface for language model implementations."""
    
    def generate(self, prompt: str, config: ModelConfig) -> GenerationResult:
        """Generate text from a prompt."""
        ...
    
    def supports_streaming(self) -> bool:
        """Check if model supports streaming."""
        ...
```

**Android Equivalent**: Like defining a repository interface that different implementations can follow (LocalRepository, RemoteRepository, etc.).

**`ITokenManager` Interface**:
```python
class ITokenManager(Protocol):
    """Interface for token management operations."""
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text."""
        ...
    
    def estimate_cost(self, tokens: int, model_name: str, is_output: bool = False) -> float:
        """Estimate cost for token usage."""
        ...
```

**Android Equivalent**: Like defining an analytics interface that different analytics providers can implement.

### 2. `exceptions.py` - The Error Hierarchy 🚨

**What it does**: Defines all custom exceptions used throughout the app - like having a clear error handling strategy.

**Key Components**:

#### Base Exception Class
```python
class LangChainAppError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
```

**Android Equivalent**: Like creating a base `AppException` class that all your custom exceptions inherit from.

#### Specific Exception Types
```python
class ModelConfigurationError(LangChainAppError):
    """Raised when model configuration is invalid."""
    pass

class ApiKeyError(LangChainAppError):
    """Raised when API key is missing or invalid."""
    pass

class GenerationError(LangChainAppError):
    """Raised when text generation fails."""
    pass

class TokenManagementError(LangChainAppError):
    """Raised when token operations fail."""
    pass
```

**Android Equivalent**: Like having specific exceptions for different error scenarios (NetworkException, DatabaseException, ValidationException).

### 3. `services.py` - The Service Implementations 🔧

**What it does**: Contains concrete implementations of core services - like your actual service classes that do the work.

**Key Components**:

#### `ApiKeyValidator` Service
```python
class ApiKeyValidator:
    """Validates API keys for different providers."""
    
    def validate_key(self, api_key: str, provider: str) -> bool:
        if not api_key:
            raise ApiKeyError(f"API key not provided for {provider}")
        
        # Check for placeholder values
        placeholder_indicators = ["your-", "sk-", "placeholder", "example", "test"]
        if any(indicator in api_key.lower() for indicator in placeholder_indicators):
            if not api_key.startswith(("sk-", "claude-", "hf_")):
                raise ApiKeyError(f"API key appears to be a placeholder for {provider}")
        
        # Provider-specific validation
        if provider.lower() == "openai" and not api_key.startswith("sk-"):
            raise ApiKeyError("OpenAI API key should start with 'sk-'")
        elif provider.lower() == "anthropic" and len(api_key) < 20:
            raise ApiKeyError("Anthropic API key appears too short")
        
        return True
```

**Android Equivalent**: Like a validation service that checks user input or API credentials before making network calls.

#### `ConfigurationManager` Service
```python
class ConfigurationManager:
    """Manages application configuration and environment variables."""
    
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self._config_cache = {}
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        
        env_var = key_mapping.get(provider.lower())
        if env_var:
            return os.getenv(env_var)
        return None
```

**Android Equivalent**: Like a configuration manager that reads from SharedPreferences, BuildConfig, or remote config.

#### `ConsoleUserInteraction` Service
```python
class ConsoleUserInteraction:
    """Handles user interaction through console interface."""
    
    def display_info(self, message: str):
        """Display informational message."""
        print(f"ℹ️  {message}")
    
    def display_warning(self, message: str):
        """Display warning message."""
        print(f"⚠️  {message}")
    
    def display_error(self, message: str):
        """Display error message."""
        print(f"❌ {message}")
    
    def get_user_input(self, prompt: str) -> str:
        """Get input from user."""
        return input(f"📝 {prompt}: ").strip()
```

**Android Equivalent**: Like a UI helper service that shows toasts, dialogs, or manages user input.

### 4. `dependency_injection.py` - The DI Container 💉

**What it does**: Manages dependency injection - like Dagger/Hilt but simpler, ensuring all services get their dependencies.

**Key Components**:

#### Container Class
```python
class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
        self._setup_default_services()
    
    def register_singleton(self, interface_type: type, implementation: Any):
        """Register a singleton service."""
        self._singletons[interface_type] = implementation
    
    def register_factory(self, interface_type: type, factory_func: callable):
        """Register a factory function for creating services."""
        self._services[interface_type] = factory_func
    
    def get(self, service_type: type):
        """Get a service instance."""
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check factories
        if service_type in self._services:
            return self._services[service_type]()
        
        raise ValueError(f"Service {service_type} not registered")
```

**Android Equivalent**: Like Dagger's component that provides dependencies, but much simpler.

#### Global Container Access
```python
_container = None

def get_container() -> DIContainer:
    """Get the global DI container."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container

def reset_container():
    """Reset the global container (useful for testing)."""
    global _container
    _container = None
```

**Android Equivalent**: Like having a global Application class that provides access to your DI component.

### 5. `utils.py` - The Helper Functions 🛠️

**What it does**: Contains utility functions used throughout the app - like your common helper methods.

**Key Components**:

#### User Interaction Helpers
```python
def prompt_continue(message: str = "Press Enter to continue...") -> bool:
    """Prompt user to continue."""
    try:
        input(f"\n{message}")
        return True
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
```

#### Model Creation Helpers
```python
def create_openai_model(api_key: str, model_name: str = "gpt-3.5-turbo"):
    """Create an OpenAI model instance."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=api_key, model=model_name)

def create_claude_model(api_key: str, model_name: str = "claude-3-sonnet-20240229"):
    """Create a Claude model instance."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(api_key=api_key, model=model_name)
```

**Android Equivalent**: Like utility classes with static helper methods (StringUtils, DateUtils, etc.).

## How to Use This Package

### Setting Up Dependency Injection
```python
from systemCore import get_container, ConfigurationManager, ApiKeyValidator

# Get the DI container
container = get_container()

# Services are automatically registered, just use them
config_manager = container.get(ConfigurationManager)
api_validator = container.get(ApiKeyValidator)
```

### Using Interfaces for Type Safety
```python
from systemCore import ILanguageModel, ModelConfig, GenerationResult

def generate_text(model: ILanguageModel, prompt: str) -> GenerationResult:
    """Generate text using any model implementation."""
    config = ModelConfig(model_name="gpt-4", temperature=0.7)
    return model.generate(prompt, config)

# This works with any model that implements ILanguageModel
result = generate_text(openai_model, "Hello world")
result = generate_text(claude_model, "Hello world")
```

### Handling Exceptions Properly
```python
from systemCore import ApiKeyError, GenerationError, ModelConfigurationError

try:
    # Some operation that might fail
    result = model.generate(prompt, config)
except ApiKeyError as e:
    print(f"API key problem: {e.message}")
    print(f"Error code: {e.error_code}")
except GenerationError as e:
    print(f"Generation failed: {e.message}")
    print(f"Details: {e.details}")
except ModelConfigurationError as e:
    print(f"Configuration error: {e.message}")
```

### Using Configuration Management
```python
from systemCore import ConfigurationManager

config_manager = ConfigurationManager()

# Get API keys
openai_key = config_manager.get_api_key("openai")
claude_key = config_manager.get_api_key("anthropic")

# Check if keys are available
if openai_key:
    print("OpenAI is configured")
else:
    print("OpenAI key not found in environment")
```

## Why This Matters for Android Developers

1. **Separation of Concerns**: Like having clear layers (data, domain, presentation)
2. **Dependency Injection**: Like Dagger/Hilt for managing dependencies
3. **Interface Segregation**: Like defining repository interfaces for testability
4. **Exception Handling**: Like having a clear error handling strategy
5. **Configuration Management**: Like BuildConfig or remote configuration
6. **Service Locator**: Like having a way to get system services

## Common Patterns You'll Recognize

- **Dependency Injection**: Container manages service creation and dependencies
- **Interface Segregation**: Small, focused interfaces for different concerns
- **Factory Pattern**: Creating different implementations of the same interface
- **Service Locator**: Global access to services through the container
- **Data Transfer Objects**: Structured data classes for passing information
- **Exception Hierarchy**: Organized error handling with specific exception types

## Files Structure
```
systemCore/
├── __init__.py                 # Package exports and main interface
├── interfaces.py              # Abstract interfaces and data classes
├── exceptions.py              # Custom exception hierarchy
├── services.py               # Concrete service implementations
├── dependency_injection.py   # DI container and service registration
├── utils.py                  # Helper functions and utilities
└── README.md                 # This documentation
```

This package is like Android's core framework - it provides the foundation that everything else builds upon, ensuring consistency, testability, and maintainability throughout your application!