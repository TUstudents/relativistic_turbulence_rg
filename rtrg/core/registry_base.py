"""
Abstract base class for field registry architecture unification.

This module provides the common interface and type system for all field
registries in the relativistic turbulence RG project, enabling type-safe
field management across different physics contexts.

Architecture:
    AbstractFieldRegistry[FieldType] - Generic base for all registries
    ├── Supports different field types (Field, TensorAwareField, SymbolicTensorField)
    ├── Defines common interface methods
    └── Enables type-safe conversions between registry types

Usage:
    >>> registry = ConcreteFieldRegistry[Field]()
    >>> field = registry.get_field("u")
    >>> all_fields = registry.get_all_fields()

The unified architecture eliminates type mismatches and manual conversions
while maintaining compatibility with existing field theory calculations.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, Union

# Generic type parameter for field types
FieldType = TypeVar("FieldType")


class AbstractFieldRegistry(ABC, Generic[FieldType]):
    """
    Abstract base class for all field registries in the RTRG project.

    Provides a unified interface for field management across different
    physics contexts (basic fields, tensor-aware fields, symbolic fields).

    The generic parameter FieldType ensures type safety when working with
    different field types while maintaining a common interface.

    Type Parameters:
        FieldType: The specific field type managed by this registry
                  (Field, TensorAwareField, SymbolicTensorField, etc.)
    """

    def __init__(self) -> None:
        """Initialize the registry with common attributes."""
        self._registry_type: str = self.__class__.__name__
        self._field_count: int = 0

    @abstractmethod
    def register_field(self, *args: Any, **kwargs: Any) -> None:
        """
        Register a field in the registry.

        Signature varies by implementation:
        - Basic: register_field(field: Field)
        - Indexed: register_field(name: str, field: SymbolicTensorField)
        """
        pass

    @abstractmethod
    def get_field(self, name: str) -> FieldType | None:
        """
        Get a field by name.

        Args:
            name: Field name identifier

        Returns:
            Field object of appropriate type, or None if not found
        """
        pass

    @abstractmethod
    def get_all_fields(self) -> dict[str, FieldType]:
        """
        Get all registered fields.

        Returns:
            Dictionary mapping field names to field objects
        """
        pass

    @abstractmethod
    def field_count(self) -> int:
        """
        Get total number of registered fields.

        Returns:
            Number of fields in registry
        """
        pass

    @abstractmethod
    def list_field_names(self) -> list[str]:
        """
        Get list of all field names.

        Returns:
            List of field name strings
        """
        pass

    def registry_type(self) -> str:
        """Get the registry type identifier."""
        return self._registry_type

    def is_empty(self) -> bool:
        """Check if registry is empty."""
        return self.field_count() == 0

    def has_field(self, name: str) -> bool:
        """Check if field exists in registry."""
        return self.get_field(name) is not None

    def __len__(self) -> int:
        """Return number of fields in registry."""
        return self.field_count()

    def __contains__(self, name: str) -> bool:
        """Support 'name in registry' syntax."""
        return self.has_field(name)

    def __str__(self) -> str:
        """String representation of registry."""
        return f"{self._registry_type}(fields={self.field_count()})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        fields = list(self.list_field_names())
        return f"{self._registry_type}(fields={fields})"


class RegistryInfo:
    """Information about a field registry for introspection and debugging."""

    def __init__(self, registry: AbstractFieldRegistry[Any]) -> None:
        self.registry_type = registry.registry_type()
        self.field_count = registry.field_count()
        self.field_names = registry.list_field_names()
        self.is_empty = registry.is_empty()

    def __str__(self) -> str:
        return (
            f"RegistryInfo({self.registry_type}, " f"{self.field_count} fields: {self.field_names})"
        )


def get_registry_info(registry: AbstractFieldRegistry[Any]) -> RegistryInfo:
    """Get detailed information about a registry."""
    return RegistryInfo(registry)


# Type aliases for common registry types (to be defined in specific modules)
BasicFieldRegistry = TypeVar("BasicFieldRegistry", bound=AbstractFieldRegistry[Any])
TensorAwareRegistry = TypeVar("TensorAwareRegistry", bound=AbstractFieldRegistry[Any])
SymbolicFieldRegistry = TypeVar("SymbolicFieldRegistry", bound=AbstractFieldRegistry[Any])
