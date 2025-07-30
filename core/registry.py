"""
Component registry for the Perspective D<cide> framework.

Manages registration and discovery of framework components.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ComponentInterface(ABC):
    """Abstract base class for framework components."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the component."""
        pass

class ComponentRegistry:
    """Registry for managing framework components."""
    
    def __init__(self):
        self._components: Dict[str, ComponentInterface] = {}
        self._factories: Dict[str, Callable] = {}
        self._config: Optional[Dict[str, Any]] = None
    
    def register_component(self, name: str, component: ComponentInterface) -> None:
        """Register a component instance."""
        self._components[name] = component
        logger.info(f"Registered component: {name}")
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a component factory function."""
        self._factories[name] = factory
        logger.info(f"Registered factory: {name}")
    
    def get_component(self, name: str) -> Optional[ComponentInterface]:
        """Get a component by name."""
        return self._components.get(name)
    
    def create_component(self, name: str, **kwargs) -> Optional[ComponentInterface]:
        """Create a component using its factory."""
        factory = self._factories.get(name)
        if factory:
            component = factory(**kwargs)
            self.register_component(name, component)
            return component
        return None
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())
    
    def initialize_all(self, config: Dict[str, Any]) -> None:
        """Initialize all registered components."""
        self._config = config
        for name, component in self._components.items():
            try:
                component.initialize(config)
                logger.info(f"Initialized component: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize component {name}: {e}")
    
    def shutdown_all(self) -> None:
        """Shutdown all registered components."""
        for name, component in self._components.items():
            try:
                component.shutdown()
                logger.info(f"Shutdown component: {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown component {name}: {e}")

# Global registry instance
_registry: Optional[ComponentRegistry] = None

def initialize(config: Dict[str, Any]) -> None:
    """Initialize the component registry."""
    global _registry
    _registry = ComponentRegistry()
    _registry.initialize_all(config)

def get_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    if _registry is None:
        raise RuntimeError("Component registry not initialized. Call initialize() first.")
    return _registry

def get_instance() -> ComponentRegistry:
    """Get the global component registry instance (alias for get_registry)."""
    return get_registry() 