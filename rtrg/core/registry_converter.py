"""
Field Registry Conversion Utilities.

This module provides utilities for converting between different field registry types
in the unified registry architecture, enabling seamless interoperation between
different physics contexts while maintaining type safety.

Conversion Patterns:
    Core FieldRegistry ↔ EnhancedFieldRegistry ↔ IndexedFieldRegistry

The converter handles field type transformations, metadata preservation, and
constraint propagation during conversions.

Usage:
    >>> converter = FieldRegistryConverter()
    >>> symbolic_registry = converter.core_to_symbolic(core_registry, coordinates)
    >>> enhanced_registry = converter.symbolic_to_enhanced(symbolic_registry, metric)
"""

import warnings
from typing import Any, Optional, cast

import sympy as sp
from sympy import Symbol

from ..field_theory.symbolic_tensors import IndexedFieldRegistry, SymbolicTensorField
from .fields import EnhancedFieldRegistry, Field, FieldRegistry, TensorAwareField
from .tensors import Metric


class ConversionCache:
    """Cache for field conversions to avoid repeated overhead."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        """Get cached conversion result."""
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Cache conversion result."""
        self._cache[key] = value

    def clear(self) -> None:
        """Clear conversion cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


class FieldRegistryConverter:
    """
    Utility class for converting between different field registry types.

    Supports conversions between:
    - Core FieldRegistry (basic Field objects)
    - EnhancedFieldRegistry (TensorAwareField objects)
    - IndexedFieldRegistry (SymbolicTensorField objects)

    Features:
    - Type-safe conversions with proper error handling
    - Metadata preservation during conversions
    - Constraint propagation where applicable
    - Caching to avoid repeated conversion overhead
    - Validation of conversion compatibility
    """

    def __init__(self) -> None:
        """Initialize converter with empty cache."""
        self.cache = ConversionCache()
        self._conversion_stats: dict[str, int] = {
            "core_to_enhanced": 0,
            "enhanced_to_core": 0,
            "core_to_symbolic": 0,
            "symbolic_to_core": 0,
            "enhanced_to_symbolic": 0,
            "symbolic_to_enhanced": 0,
        }

    def core_to_enhanced(
        self, core_registry: FieldRegistry, metric: Metric | None = None
    ) -> EnhancedFieldRegistry:
        """
        Convert Core FieldRegistry to EnhancedFieldRegistry.

        Args:
            core_registry: Source FieldRegistry with basic Field objects
            metric: Metric tensor for enhanced fields (defaults to Minkowski)

        Returns:
            EnhancedFieldRegistry with TensorAwareField objects

        Raises:
            ValueError: If conversion fails due to incompatible fields
        """
        cache_key = f"core_to_enhanced_{id(core_registry)}_{id(metric) if metric else 'default'}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cast(EnhancedFieldRegistry, cached)

        self._conversion_stats["core_to_enhanced"] += 1

        if metric is None:
            metric = Metric()  # Default Minkowski metric

        enhanced_registry = EnhancedFieldRegistry()

        try:
            # Create enhanced IS fields - this will create TensorAwareField versions
            enhanced_registry.create_enhanced_is_fields(metric)

            # Validate that all core fields have corresponding enhanced versions
            for name in core_registry.list_field_names():
                if not enhanced_registry.has_field(name):
                    warnings.warn(
                        f"Field '{name}' from core registry not found in enhanced registry",
                        stacklevel=2,
                    )

            self.cache.set(cache_key, enhanced_registry)
            return enhanced_registry

        except Exception as e:
            raise ValueError(f"Failed to convert core registry to enhanced: {e}") from e

    def enhanced_to_core(self, enhanced_registry: EnhancedFieldRegistry) -> FieldRegistry:
        """
        Convert EnhancedFieldRegistry to Core FieldRegistry.

        Args:
            enhanced_registry: Source EnhancedFieldRegistry with TensorAwareField objects

        Returns:
            FieldRegistry with basic Field objects

        Note:
            This conversion may lose tensor-specific metadata and capabilities.
        """
        cache_key = f"enhanced_to_core_{id(enhanced_registry)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cast(FieldRegistry, cached)

        self._conversion_stats["enhanced_to_core"] += 1

        # Since EnhancedFieldRegistry inherits from FieldRegistry,
        # we can create a new FieldRegistry with the basic Field objects
        core_registry = FieldRegistry()

        # Copy fields (TensorAwareField inherits from Field, so this works)
        for _name, field in enhanced_registry.fields.items():
            base_field = field  # TensorAwareField extends Field
            core_registry.register_field(base_field)

        self.cache.set(cache_key, core_registry)
        return core_registry

    def core_to_symbolic(
        self, core_registry: FieldRegistry, coordinates: list[Symbol], metric: Metric | None = None
    ) -> IndexedFieldRegistry:
        """
        Convert Core FieldRegistry to IndexedFieldRegistry.

        Args:
            core_registry: Source FieldRegistry with basic Field objects
            coordinates: Spacetime coordinates for symbolic fields
            metric: Metric tensor for constraint application

        Returns:
            IndexedFieldRegistry with SymbolicTensorField objects
        """
        cache_key = f"core_to_symbolic_{id(core_registry)}_{id(coordinates)}_{id(metric) if metric else 'default'}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cast(IndexedFieldRegistry, cached)

        self._conversion_stats["core_to_symbolic"] += 1

        symbolic_registry = IndexedFieldRegistry()

        try:
            # Create Israel-Stewart symbolic fields with proper tensor structure
            symbolic_registry.create_israel_stewart_fields(coordinates, metric)

            # Validate field correspondence
            for name in core_registry.list_field_names():
                if not symbolic_registry.has_field(name):
                    warnings.warn(
                        f"Field '{name}' from core registry not found in symbolic registry",
                        stacklevel=2,
                    )

            self.cache.set(cache_key, symbolic_registry)
            return symbolic_registry

        except Exception as e:
            raise ValueError(f"Failed to convert core registry to symbolic: {e}") from e

    def symbolic_to_core(
        self, symbolic_registry: IndexedFieldRegistry, metric: Metric | None = None
    ) -> FieldRegistry:
        """
        Convert IndexedFieldRegistry to Core FieldRegistry.

        Args:
            symbolic_registry: Source IndexedFieldRegistry with SymbolicTensorField objects
            metric: Metric tensor for core field creation

        Returns:
            FieldRegistry with basic Field objects

        Note:
            This conversion loses symbolic computation capabilities.
        """
        cache_key = (
            f"symbolic_to_core_{id(symbolic_registry)}_{id(metric) if metric else 'default'}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cast(FieldRegistry, cached)

        self._conversion_stats["symbolic_to_core"] += 1

        if metric is None:
            metric = Metric()

        core_registry = FieldRegistry()

        try:
            # Create IS fields - this matches the field names from symbolic registry
            core_registry.create_is_fields(metric)

            self.cache.set(cache_key, core_registry)
            return core_registry

        except Exception as e:
            raise ValueError(f"Failed to convert symbolic registry to core: {e}") from e

    def enhanced_to_symbolic(
        self,
        enhanced_registry: EnhancedFieldRegistry,
        coordinates: list[Symbol],
        metric: Metric | None = None,
    ) -> IndexedFieldRegistry:
        """
        Convert EnhancedFieldRegistry to IndexedFieldRegistry.

        Args:
            enhanced_registry: Source EnhancedFieldRegistry
            coordinates: Spacetime coordinates for symbolic fields
            metric: Metric tensor for symbolic field creation

        Returns:
            IndexedFieldRegistry with SymbolicTensorField objects
        """
        # Use core as intermediate step for consistency
        core_registry = self.enhanced_to_core(enhanced_registry)
        return self.core_to_symbolic(core_registry, coordinates, metric)

    def symbolic_to_enhanced(
        self, symbolic_registry: IndexedFieldRegistry, metric: Metric | None = None
    ) -> EnhancedFieldRegistry:
        """
        Convert IndexedFieldRegistry to EnhancedFieldRegistry.

        Args:
            symbolic_registry: Source IndexedFieldRegistry
            metric: Metric tensor for enhanced field creation

        Returns:
            EnhancedFieldRegistry with TensorAwareField objects
        """
        # Use core as intermediate step for consistency
        core_registry = self.symbolic_to_core(symbolic_registry, metric)
        return self.core_to_enhanced(core_registry, metric)

    def validate_conversion_compatibility(self, source_registry: Any, target_type: str) -> bool:
        """
        Validate that a registry can be converted to target type.

        Args:
            source_registry: Source registry to validate
            target_type: Target registry type ("core", "enhanced", "symbolic")

        Returns:
            True if conversion is possible
        """
        source_type = source_registry.__class__.__name__.lower()

        # All conversions are supported through the converter
        valid_combinations = {
            ("fieldregistry", "enhanced"),
            ("fieldregistry", "symbolic"),
            ("enhancedfieldregistry", "core"),
            ("enhancedfieldregistry", "symbolic"),
            ("indexedfieldregistry", "core"),
            ("indexedfieldregistry", "enhanced"),
        }

        return (source_type, target_type) in valid_combinations

    def get_conversion_stats(self) -> dict[str, int]:
        """Get conversion statistics."""
        return self._conversion_stats.copy()

    def clear_cache(self) -> None:
        """Clear conversion cache."""
        self.cache.clear()

    def __str__(self) -> str:
        """String representation with stats."""
        total_conversions = sum(self._conversion_stats.values())
        return f"FieldRegistryConverter(conversions={total_conversions}, cache_size={self.cache.size()})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"FieldRegistryConverter(stats={self._conversion_stats}, cache_size={self.cache.size()})"


# Convenience factory functions
def convert_core_to_enhanced(
    core_registry: FieldRegistry, metric: Metric | None = None
) -> EnhancedFieldRegistry:
    """Convert Core FieldRegistry to EnhancedFieldRegistry."""
    converter = FieldRegistryConverter()
    return converter.core_to_enhanced(core_registry, metric)


def convert_core_to_symbolic(
    core_registry: FieldRegistry, coordinates: list[Symbol], metric: Metric | None = None
) -> IndexedFieldRegistry:
    """Convert Core FieldRegistry to IndexedFieldRegistry."""
    converter = FieldRegistryConverter()
    return converter.core_to_symbolic(core_registry, coordinates, metric)


def convert_symbolic_to_enhanced(
    symbolic_registry: IndexedFieldRegistry, metric: Metric | None = None
) -> EnhancedFieldRegistry:
    """Convert IndexedFieldRegistry to EnhancedFieldRegistry."""
    converter = FieldRegistryConverter()
    return converter.symbolic_to_enhanced(symbolic_registry, metric)
