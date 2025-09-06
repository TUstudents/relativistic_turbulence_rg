"""
Registry Factory for Context-Aware Field Registry Selection.

This module provides a factory system for automatically selecting and creating
the appropriate field registry type based on the physics context, eliminating
manual registry instantiation and conversion throughout the codebase.

Factory Patterns:
    RegistryFactory.create_for_context(context) -> Appropriate registry type
    RegistryFactory.auto_detect_context(module, operation) -> Context string

Supported Contexts:
    - "basic_physics": Core FieldRegistry for simple IS calculations
    - "tensor_operations": EnhancedFieldRegistry for advanced tensor algebra
    - "symbolic_msrjd": IndexedFieldRegistry for symbolic field theory
    - "auto": Automatic context detection based on usage patterns

Usage:
    >>> factory = RegistryFactory()
    >>> registry = factory.create_for_context("tensor_operations", metric=metric)
    >>> registry = factory.create_for_module("propagators")  # Auto-detects context
"""

import inspect
import warnings
from enum import Enum
from typing import Any, Optional, Union, cast

import sympy as sp
from sympy import Symbol

from ..field_theory.symbolic_tensors import IndexedFieldRegistry, SymbolicTensorField
from .fields import EnhancedFieldRegistry, Field, FieldRegistry
from .registry_base import AbstractFieldRegistry
from .registry_converter import FieldRegistryConverter
from .tensors import Metric


class RegistryContext(Enum):
    """Enumeration of supported registry contexts."""

    BASIC_PHYSICS = "basic_physics"
    TENSOR_OPERATIONS = "tensor_operations"
    SYMBOLIC_MSRJD = "symbolic_msrjd"
    AUTO = "auto"


class ContextDetector:
    """Automatic context detection based on module usage patterns."""

    def __init__(self) -> None:
        """Initialize context detector with pattern rules."""
        self._module_patterns = {
            # Basic physics patterns
            "basic_physics": {
                "modules": ["israel_stewart.equations", "core.fields"],
                "operations": ["create_is_fields", "register_field", "get_field"],
                "keywords": ["IsraelStewartSystem", "basic", "simple"],
            },
            # Tensor operations patterns
            "tensor_operations": {
                "modules": ["field_theory.propagators", "core.tensors"],
                "operations": ["construct_tensor_aware_propagator_matrix", "validate_constraints"],
                "keywords": ["TensorAware", "Enhanced", "constraint", "projector"],
            },
            # Symbolic MSRJD patterns
            "symbolic_msrjd": {
                "modules": ["field_theory.tensor_msrjd_action", "field_theory.symbolic_tensors"],
                "operations": [
                    "construct_action",
                    "create_antifield",
                    "generate_field_action_pairs",
                ],
                "keywords": ["symbolic", "MSRJD", "action", "antifield", "SymbolicTensorField"],
            },
        }

    def detect_from_module_name(self, module_name: str) -> str:
        """Detect context from module name."""
        for context, patterns in self._module_patterns.items():
            for module_pattern in patterns["modules"]:
                if module_pattern in module_name:
                    return context
        return "basic_physics"  # Default fallback

    def detect_from_operation(self, operation_name: str) -> str:
        """Detect context from operation name."""
        for context, patterns in self._module_patterns.items():
            for op_pattern in patterns["operations"]:
                if op_pattern in operation_name:
                    return context
        return "basic_physics"  # Default fallback

    def detect_from_keywords(self, keywords: list[str]) -> str:
        """Detect context from a list of keywords."""
        context_scores: dict[str, int] = {
            "basic_physics": 0,
            "tensor_operations": 0,
            "symbolic_msrjd": 0,
        }

        for keyword in keywords:
            for context, patterns in self._module_patterns.items():
                for pattern_keyword in patterns["keywords"]:
                    if pattern_keyword.lower() in keyword.lower():
                        context_scores[context] += 1

        # Return context with highest score
        best_context = max(context_scores.items(), key=lambda x: x[1])
        return best_context[0] if best_context[1] > 0 else "basic_physics"

    def detect_from_caller_stack(self) -> str:
        """Detect context from calling stack analysis."""
        stack = inspect.stack()

        # Look at calling module names
        for frame_info in stack[2:6]:  # Skip factory frames, look at callers
            filename = frame_info.filename
            if "rtrg" in filename:
                # Extract module path from filename
                module_parts = filename.split("rtrg")[-1].replace("/", ".").replace(".py", "")
                return self.detect_from_module_name(module_parts)

        return "basic_physics"  # Default fallback


class RegistryFactory:
    """
    Factory for creating context-appropriate field registries.

    Automatically selects the best registry type based on physics context,
    eliminating manual registry selection and conversion throughout the codebase.

    Features:
    - Context-aware registry selection
    - Automatic parameter setup (metrics, coordinates)
    - Registry conversion when needed
    - Performance optimization through caching
    - Usage analytics for optimization
    """

    def __init__(self) -> None:
        """Initialize factory with context detector and converter."""
        self.context_detector = ContextDetector()
        self.converter = FieldRegistryConverter()
        self._creation_stats: dict[str, int] = {
            "basic_physics": 0,
            "tensor_operations": 0,
            "symbolic_msrjd": 0,
            "auto_detected": 0,
        }
        self._registry_cache: dict[str, Any] = {}

    def create_for_context(
        self, context: str | RegistryContext, **kwargs: Any
    ) -> AbstractFieldRegistry[Any]:
        """
        Create registry for specific context.

        Args:
            context: Registry context ("basic_physics", "tensor_operations", "symbolic_msrjd", "auto")
            **kwargs: Context-specific parameters
                - metric: Metric tensor (for enhanced/symbolic)
                - coordinates: Spacetime coordinates (for symbolic)

        Returns:
            Appropriate registry type for the context

        Raises:
            ValueError: If context is invalid or required parameters missing
        """
        if isinstance(context, RegistryContext):
            context = context.value

        if context == "auto":
            context = self.context_detector.detect_from_caller_stack()
            self._creation_stats["auto_detected"] += 1

        self._creation_stats[context] += 1

        # Create cache key (handle unhashable values like lists)
        hashable_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                hashable_kwargs[k] = tuple(str(item) for item in v)
            elif hasattr(v, "__hash__") and callable(v.__hash__):
                hashable_kwargs[k] = v
            else:
                hashable_kwargs[k] = str(v)
        cache_key = f"{context}_{hash(frozenset(hashable_kwargs.items()))}"
        if cache_key in self._registry_cache:
            return cast(AbstractFieldRegistry[Any], self._registry_cache[cache_key])

        registry = self._create_registry_for_context(context, **kwargs)
        self._registry_cache[cache_key] = registry
        return registry

    def _create_registry_for_context(
        self, context: str, **kwargs: Any
    ) -> AbstractFieldRegistry[Any]:
        """Create registry for specific context with parameters."""
        if context == "basic_physics":
            registry = FieldRegistry()
            metric = kwargs.get("metric")
            registry.create_is_fields(metric)
            return registry

        elif context == "tensor_operations":
            registry = EnhancedFieldRegistry()
            metric = kwargs.get("metric") or Metric()
            registry.create_enhanced_is_fields(metric)
            return registry

        elif context == "symbolic_msrjd":
            symbolic_registry = IndexedFieldRegistry()
            coordinates = kwargs.get("coordinates")
            if coordinates is None:
                # Create default spacetime coordinates
                coordinates = [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]
            metric = kwargs.get("metric")
            symbolic_registry.create_israel_stewart_fields(coordinates, metric)
            return symbolic_registry

        else:
            raise ValueError(f"Unknown registry context: {context}")

    def create_for_module(self, module_name: str, **kwargs: Any) -> AbstractFieldRegistry[Any]:
        """
        Create registry based on module usage context.

        Args:
            module_name: Name of the calling module
            **kwargs: Additional parameters for registry creation

        Returns:
            Appropriate registry for the module's needs
        """
        context = self.context_detector.detect_from_module_name(module_name)
        return self.create_for_context(context, **kwargs)

    def create_for_operation(
        self, operation_name: str, **kwargs: Any
    ) -> AbstractFieldRegistry[Any]:
        """
        Create registry based on operation context.

        Args:
            operation_name: Name of the operation being performed
            **kwargs: Additional parameters for registry creation

        Returns:
            Appropriate registry for the operation's needs
        """
        context = self.context_detector.detect_from_operation(operation_name)
        return self.create_for_context(context, **kwargs)

    def create_auto(self, **kwargs: Any) -> AbstractFieldRegistry[Any]:
        """
        Automatically create registry based on calling context.

        Args:
            **kwargs: Parameters for registry creation

        Returns:
            Registry automatically selected based on usage context
        """
        return self.create_for_context("auto", **kwargs)

    def convert_registry(
        self,
        source_registry: AbstractFieldRegistry[Any],
        target_context: str | RegistryContext,
        **kwargs: Any,
    ) -> AbstractFieldRegistry[Any]:
        """
        Convert existing registry to different context.

        Args:
            source_registry: Existing registry to convert
            target_context: Target registry context
            **kwargs: Parameters for target registry

        Returns:
            Registry converted to target context
        """
        if isinstance(target_context, RegistryContext):
            target_context = target_context.value

        source_type = source_registry.__class__.__name__

        # Use converter for registry transformation
        if target_context == "basic_physics":
            if "Enhanced" in source_type:
                return self.converter.enhanced_to_core(cast(EnhancedFieldRegistry, source_registry))
            elif "Indexed" in source_type:
                return self.converter.symbolic_to_core(
                    cast(IndexedFieldRegistry, source_registry), kwargs.get("metric")
                )

        elif target_context == "tensor_operations":
            if "FieldRegistry" in source_type and "Enhanced" not in source_type:
                return self.converter.core_to_enhanced(
                    cast(FieldRegistry, source_registry), kwargs.get("metric")
                )
            elif "Indexed" in source_type:
                return self.converter.symbolic_to_enhanced(
                    cast(IndexedFieldRegistry, source_registry), kwargs.get("metric")
                )

        elif target_context == "symbolic_msrjd":
            coordinates = kwargs.get(
                "coordinates", [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]
            )
            if "FieldRegistry" in source_type and "Enhanced" not in source_type:
                return self.converter.core_to_symbolic(
                    cast(FieldRegistry, source_registry), coordinates, kwargs.get("metric")
                )
            elif "Enhanced" in source_type:
                return self.converter.enhanced_to_symbolic(
                    cast(EnhancedFieldRegistry, source_registry), coordinates, kwargs.get("metric")
                )

        # If no conversion needed, return as-is
        return source_registry

    def get_recommended_context(self, **hints: Any) -> str:
        """
        Get recommended registry context based on usage hints.

        Args:
            **hints: Usage hints
                - module: Module name
                - operation: Operation name
                - keywords: List of relevant keywords

        Returns:
            Recommended context string
        """
        if "module" in hints:
            return self.context_detector.detect_from_module_name(hints["module"])
        if "operation" in hints:
            return self.context_detector.detect_from_operation(hints["operation"])
        if "keywords" in hints:
            return self.context_detector.detect_from_keywords(hints["keywords"])

        return self.context_detector.detect_from_caller_stack()

    def get_creation_stats(self) -> dict[str, int]:
        """Get registry creation statistics."""
        return self._creation_stats.copy()

    def clear_cache(self) -> None:
        """Clear registry cache."""
        self._registry_cache.clear()
        self.converter.clear_cache()

    def __str__(self) -> str:
        """String representation with stats."""
        total_created = sum(self._creation_stats.values())
        return f"RegistryFactory(created={total_created}, cached={len(self._registry_cache)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"RegistryFactory(stats={self._creation_stats}, cache_size={len(self._registry_cache)})"
        )


# Global factory instance for convenience
_global_factory = RegistryFactory()


def create_registry_for_context(
    context: str | RegistryContext, **kwargs: Any
) -> AbstractFieldRegistry[Any]:
    """Create registry using global factory."""
    return _global_factory.create_for_context(context, **kwargs)


def create_auto_registry(**kwargs: Any) -> AbstractFieldRegistry[Any]:
    """Automatically create registry based on context."""
    return _global_factory.create_auto(**kwargs)


def get_registry_factory() -> RegistryFactory:
    """Get global registry factory instance."""
    return _global_factory
