"""
Dependency injection container for the LangChain application.
Manages service creation and dependency resolution.
"""

from typing import Dict, Any, TypeVar, Type, Optional
from .interfaces import ITokenManager, IUserInteraction
from .services import ConfigurationManager, ApiKeyValidator, ConsoleUserInteraction, LoggingService
from .token_utils import TokenManager
from .models.model_factory import ModelFactory

T = TypeVar('T')


class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._setup_default_services()
    
    def _setup_default_services(self) -> None:
        """Setup default service registrations."""
        # Configuration
        self.register_singleton(ConfigurationManager, ConfigurationManager)
        
        # Logging (depends on configuration)
        self.register_factory(LoggingService, lambda: LoggingService(self.get(ConfigurationManager)))
        
        # Services
        self.register_singleton(ApiKeyValidator, ApiKeyValidator)
        self.register_singleton(TokenManager, TokenManager)
        
        # User interaction (depends on logging)
        self.register_factory(
            ConsoleUserInteraction, 
            lambda: ConsoleUserInteraction(self.get(LoggingService).get_logger("user_interaction"))
        )
        
        # Model factory (depends on multiple services)
        self.register_factory(
            ModelFactory,
            lambda: ModelFactory(
                config_manager=self.get(ConfigurationManager),
                api_key_validator=self.get(ApiKeyValidator),
                token_manager=self.get(TokenManager),
                user_interaction=self.get(ConsoleUserInteraction),
                logging_service=self.get(LoggingService)
            )
        )
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._services[interface.__name__] = ('singleton', implementation)
    
    def register_factory(self, interface: Type[T], factory_func) -> None:
        """Register a factory function for creating services."""
        self._services[interface.__name__] = ('factory', factory_func)
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance."""
        self._singletons[interface.__name__] = instance
    
    def get(self, interface: Type[T]) -> T:
        """Get a service instance."""
        service_name = interface.__name__
        
        # Check if already instantiated as singleton
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check if registered
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        
        service_type, service_impl = self._services[service_name]
        
        if service_type == 'singleton':
            # Create singleton instance
            instance = service_impl()
            self._singletons[service_name] = instance
            return instance
        elif service_type == 'factory':
            # Call factory function
            return service_impl()
        else:
            raise ValueError(f"Unknown service type: {service_type}")
    
    def get_token_manager(self) -> ITokenManager:
        """Get token manager service."""
        return self.get(TokenManager)
    
    def get_user_interaction(self) -> IUserInteraction:
        """Get user interaction service."""
        return self.get(ConsoleUserInteraction)
    
    def get_model_factory(self) -> ModelFactory:
        """Get model factory service."""
        return self.get(ModelFactory)
    
    def get_config_manager(self) -> ConfigurationManager:
        """Get configuration manager service."""
        return self.get(ConfigurationManager)
    
    def get_logging_service(self) -> LoggingService:
        """Get logging service."""
        return self.get(LoggingService)


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _container
    _container = None