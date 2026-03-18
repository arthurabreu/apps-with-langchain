"""
Dependency injection container for the LangChain application.

Android Analogy: 
- This is your Dagger/Hilt/Koin setup. 
- DIContainer is the 'Component' or 'Module'.
- It manages lifetimes (Singletons vs Factories).
"""

from typing import Dict, Any, TypeVar, Type, Optional
from .interfaces import ITokenManager, IUserInteraction, IFileExporter
from .services import ConfigurationManager, ApiKeyValidator, ConsoleUserInteraction, LoggingService
from .token_utils import TokenManager
from .models.model_factory import ModelFactory
from .prompt_manager import PromptManager
from .cost_tracker import CostTracker
from .exporters import ExcelExporter
from .cli_service import InteractiveCLI

T = TypeVar('T')


class DIContainer:
    """
    Simple dependency injection container.
    
    Android Analogy: A manual DI implementation or a Koin 'module'.
    It uses a Map (dict) to store service providers.
    """
    
    def __init__(self):
        """Initialize the container and setup default bindings."""
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._setup_default_services()
    
    def _setup_default_services(self) -> None:
        """
        Setup default service registrations.
        Equivalent to Hilt's @Provides or Dagger's @Module methods.
        """
        # Configuration
        self.register_singleton(ConfigurationManager, ConfigurationManager)

        # Logging (depends on configuration)
        # Lambda here is like a Provider<T> or a Factory in Kotlin.
        self.register_factory(LoggingService, lambda: LoggingService(self.get(ConfigurationManager)))

        # Services
        self.register_singleton(ApiKeyValidator, ApiKeyValidator)
        self.register_singleton(TokenManager, TokenManager)
        self.register_singleton(PromptManager, PromptManager)
        self.register_singleton(CostTracker, CostTracker)
        self.register_singleton(ExcelExporter, ExcelExporter)
        self.register_factory(InteractiveCLI, lambda: InteractiveCLI(self))

        # User interaction (depends on logging)
        self.register_factory(
            ConsoleUserInteraction,
            lambda: ConsoleUserInteraction(self.get(LoggingService).get_logger("user_interaction"))
        )

        # Model factory (complex graph of dependencies)
        self.register_factory(
            ModelFactory,
            lambda: ModelFactory(
                config_manager=self.get(ConfigurationManager),
                api_key_validator=self.get(ApiKeyValidator),
                token_manager=self.get(TokenManager),
                user_interaction=self.get(ConsoleUserInteraction),
                logging_service=self.get(LoggingService),
                cost_tracker=self.get(CostTracker)
            )
        )
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """
        Register a singleton service.
        Equivalent to @Singleton in Hilt/Dagger.
        """
        self._services[interface.__name__] = ('singleton', implementation)
    
    def register_factory(self, interface: Type[T], factory_func) -> None:
        """
        Register a factory function for creating services.
        Equivalent to a factory that creates a new instance every time.
        """
        self._services[interface.__name__] = ('factory', factory_func)
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register a specific already-created instance.
        Equivalent to 'bindInstance' in some DI frameworks.
        """
        self._singletons[interface.__name__] = instance
    
    def get(self, interface: Type[T]) -> T:
        """
        Get a service instance.
        Equivalent to 'get()' in Koin or 'inject()' in Dagger.
        
        Args:
            interface: The class or interface type you want to retrieve.
            
        Returns:
            The resolved instance.
        """
        service_name = interface.__name__
        
        # Check if already instantiated as singleton
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check if registered
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        
        service_type, service_impl = self._services[service_name]
        
        if service_type == 'singleton':
            # Create singleton instance once
            instance = service_impl()
            self._singletons[service_name] = instance
            return instance
        elif service_type == 'factory':
            # Call factory function to get a new instance
            return service_impl()
        else:
            raise ValueError(f"Unknown service type: {service_type}")
    
    def get_token_manager(self) -> ITokenManager:
        """Convenience getter for TokenManager."""
        return self.get(TokenManager)
    
    def get_user_interaction(self) -> IUserInteraction:
        """Convenience getter for UI interaction."""
        return self.get(ConsoleUserInteraction)
    
    def get_model_factory(self) -> ModelFactory:
        """Convenience getter for ModelFactory."""
        return self.get(ModelFactory)
    
    def get_config_manager(self) -> ConfigurationManager:
        """Convenience getter for ConfigManager."""
        return self.get(ConfigurationManager)
    
    def get_logging_service(self) -> LoggingService:
        """Convenience getter for LoggingService."""
        return self.get(LoggingService)

    def get_prompt_manager(self) -> PromptManager:
        """Convenience getter for PromptManager."""
        return self.get(PromptManager)

    def get_cost_tracker(self) -> CostTracker:
        """Convenience getter for CostTracker."""
        return self.get(CostTracker)

    def get_file_exporter(self) -> IFileExporter:
        """Convenience getter for FileExporter."""
        return self.get(ExcelExporter)

    def get_cli_service(self) -> InteractiveCLI:
        """Convenience getter for CLI orchestrator."""
        return self.get(InteractiveCLI)


# Global container instance (Singleton of the Container itself)
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """
    Get the global DI container instance.
    Lazy initialization (Thread-safe check omitted for simplicity).
    """
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def reset_container() -> None:
    """Reset the global container (useful for Unit Testing)."""
    global _container
    _container = None