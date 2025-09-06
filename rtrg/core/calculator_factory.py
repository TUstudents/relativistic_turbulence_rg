"""
Calculator Factory for Context-Aware Calculator Creation.

This module provides a factory system for creating the appropriate calculator
instances based on physics context and requirements, following the same
pattern as the registry factory for consistent architecture.

Factory Patterns:
    CalculatorFactory.create_propagator_calculator(context) -> PropagatorCalculatorBase

Supported Contexts:
    - "simple": Basic propagator calculations without tensor handling
    - "enhanced": Advanced calculations with full feature set
    - "tensor_aware": Full tensor index handling and constraints

Usage:
    >>> factory = CalculatorFactory()
    >>> calc = factory.create_propagator_calculator("simple", msrjd_action=action)
    >>> calc = create_propagator_calculator("enhanced", msrjd_action=action, metric=metric)
"""

import warnings
from enum import Enum
from typing import Any, Optional, cast

from ..field_theory.msrjd_action import MSRJDAction
from .calculator_implementations import SimplePropagatorCalculator
from .calculators import AbstractCalculator, PropagatorCalculatorBase


class CalculatorContext(Enum):
    """Enumeration of supported calculator contexts."""

    SIMPLE = "simple"
    ENHANCED = "enhanced"
    TENSOR_AWARE = "tensor_aware"


class CalculatorFactory:
    """
    Factory for creating context-appropriate calculators.

    Provides explicit context-based calculator selection, eliminating manual
    calculator instantiation throughout the codebase and ensuring consistent
    parameter setup.

    Features:
    - Explicit context-aware calculator selection
    - Automatic parameter validation and setup
    - Performance optimization through caching
    - Usage analytics for optimization
    """

    def __init__(self) -> None:
        """Initialize calculator factory."""
        self._creation_stats: dict[str, int] = {
            "simple": 0,
            "enhanced": 0,
            "tensor_aware": 0,
        }
        self._calculator_cache: dict[str, Any] = {}

    def create_propagator_calculator(
        self, context: str | CalculatorContext, msrjd_action: MSRJDAction, **kwargs: Any
    ) -> PropagatorCalculatorBase:
        """
        Create propagator calculator for specific context.

        Args:
            context: Calculator context ("simple", "enhanced", "tensor_aware")
            msrjd_action: MSRJD action instance for propagator calculations
            **kwargs: Context-specific parameters
                - metric: Metric tensor (for enhanced/tensor_aware)
                - temperature: System temperature
                - enable_caching: Whether to enable result caching

        Returns:
            Appropriate propagator calculator for the context

        Raises:
            ValueError: If context is invalid or required parameters missing
        """
        if isinstance(context, CalculatorContext):
            context = context.value

        if context not in ["simple", "enhanced", "tensor_aware"]:
            raise ValueError(
                f"Invalid calculator context: {context}. Must be one of: simple, enhanced, tensor_aware"
            )

        self._creation_stats[context] += 1

        # Create cache key
        cache_key = self._create_cache_key(context, msrjd_action, **kwargs)
        if cache_key in self._calculator_cache:
            return cast(PropagatorCalculatorBase, self._calculator_cache[cache_key])

        calculator = self._create_propagator_calculator_for_context(context, msrjd_action, **kwargs)
        self._calculator_cache[cache_key] = calculator
        return calculator

    def _create_cache_key(self, context: str, msrjd_action: MSRJDAction, **kwargs: Any) -> str:
        """Create cache key for calculator instance."""
        # Use action object id and context as base
        base_key = f"{context}_{id(msrjd_action)}"

        # Add parameter-specific suffixes
        if "temperature" in kwargs:
            base_key += f"_T_{kwargs['temperature']}"
        if "enable_caching" in kwargs:
            base_key += f"_cache_{kwargs['enable_caching']}"
        if "metric" in kwargs and kwargs["metric"] is not None:
            base_key += f"_metric_{id(kwargs['metric'])}"

        return base_key

    def _create_propagator_calculator_for_context(
        self, context: str, msrjd_action: MSRJDAction, **kwargs: Any
    ) -> PropagatorCalculatorBase:
        """Create propagator calculator for specific context with parameters."""

        if context == "simple":
            temperature = kwargs.get("temperature", 1.0)
            enable_caching = kwargs.get("enable_caching", True)

            return SimplePropagatorCalculator(
                msrjd_action=msrjd_action, temperature=temperature, enable_caching=enable_caching
            )

        elif context == "enhanced":
            # Import here to avoid circular dependencies
            try:
                from ..field_theory.propagators import (
                    PropagatorCalculator as LegacyPropagatorCalculator,
                )

                metric = kwargs.get("metric")
                temperature = kwargs.get("temperature", 1.0)

                # Create legacy calculator instance
                # This will be replaced with EnhancedPropagatorCalculator in next phase
                legacy_calc = LegacyPropagatorCalculator(
                    msrjd_action=msrjd_action, metric=metric, temperature=temperature
                )

                # Wrap in adapter to provide unified interface
                return PropagatorCalculatorAdapter(legacy_calc, "enhanced")

            except ImportError as e:
                warnings.warn(f"Could not import enhanced propagator calculator: {str(e)}")
                # Fallback to simple calculator
                return self._create_propagator_calculator_for_context(
                    "simple", msrjd_action, **kwargs
                )

        elif context == "tensor_aware":
            try:
                from ..field_theory.propagators import TensorAwarePropagatorCalculator

                metric = kwargs.get("metric")
                temperature = kwargs.get("temperature", 1.0)

                if metric is None:
                    from ..core.tensors import Metric

                    metric = Metric()

                tensor_calc = TensorAwarePropagatorCalculator(
                    msrjd_action=msrjd_action, metric=metric, temperature=temperature
                )

                # Wrap in adapter to provide unified interface
                return PropagatorCalculatorAdapter(tensor_calc, "tensor_aware")

            except ImportError as e:
                warnings.warn(f"Could not import tensor-aware propagator calculator: {str(e)}")
                # Fallback to enhanced or simple
                return self._create_propagator_calculator_for_context(
                    "enhanced", msrjd_action, **kwargs
                )

        else:
            raise ValueError(f"Unknown calculator context: {context}")

    def get_creation_stats(self) -> dict[str, int]:
        """Get calculator creation statistics."""
        return self._creation_stats.copy()

    def clear_cache(self) -> None:
        """Clear calculator cache."""
        self._calculator_cache.clear()

    def __str__(self) -> str:
        """String representation with stats."""
        total_created = sum(self._creation_stats.values())
        return f"CalculatorFactory(created={total_created}, cached={len(self._calculator_cache)})"


class PropagatorCalculatorAdapter(PropagatorCalculatorBase):
    """
    Adapter class for legacy propagator calculators.

    Provides unified interface for existing calculator implementations
    while maintaining backward compatibility during transition period.
    """

    def __init__(self, legacy_calculator: Any, context: str):
        """
        Initialize adapter with legacy calculator.

        Args:
            legacy_calculator: Existing calculator instance
            context: Context type for proper interface mapping
        """
        # Initialize base with legacy calculator's parameters
        msrjd_action = (
            legacy_calculator.action
            if hasattr(legacy_calculator, "action")
            else legacy_calculator.msrjd_action
        )
        metric = getattr(legacy_calculator, "metric", None)
        temperature = getattr(legacy_calculator, "temperature", 1.0)

        super().__init__(msrjd_action, metric, temperature)

        self._legacy_calc = legacy_calculator
        self._context = context

        # Expose legacy calculator attributes for backward compatibility
        self.action = legacy_calculator
        self.propagator_cache = getattr(legacy_calculator, "propagator_cache", {})
        self.matrix_cache = getattr(legacy_calculator, "matrix_cache", {})
        self.quadratic_action = getattr(legacy_calculator, "quadratic_action", None)

    def calculate_retarded_propagator(
        self,
        field1: Any,
        field2: Any,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> Any:
        """Delegate to legacy calculator's retarded propagator method."""
        try:
            return self._legacy_calc.calculate_retarded_propagator(field1, field2, omega_val, k_val)
        except Exception as e:
            warnings.warn(f"Legacy calculator failed: {str(e)}")
            # Return simple default
            return 1 / (-1j * (omega_val or self.omega) + (k_val or self.k) ** 2)

    def calculate_advanced_propagator(
        self,
        field1: Any,
        field2: Any,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> Any:
        """Delegate to legacy calculator's advanced propagator method."""
        try:
            return self._legacy_calc.calculate_advanced_propagator(field1, field2, omega_val, k_val)
        except Exception as e:
            warnings.warn(f"Legacy calculator failed: {str(e)}")
            # Fallback: use retarded with advanced prescription
            retarded = self.calculate_retarded_propagator(field1, field2, omega_val, k_val)
            import sympy as sp

            return sp.conjugate(retarded.subs(self.omega, -sp.conjugate(self.omega)))

    def calculate_keldysh_propagator(
        self,
        field1: Any,
        field2: Any,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> Any:
        """Delegate to legacy calculator's Keldysh propagator method."""
        try:
            return self._legacy_calc.calculate_keldysh_propagator(field1, field2, omega_val, k_val)
        except Exception as e:
            warnings.warn(f"Legacy calculator failed: {str(e)}")
            # Fallback: use FDT relation
            retarded = self.calculate_retarded_propagator(field1, field2, omega_val, k_val)
            advanced = self.calculate_advanced_propagator(field1, field2, omega_val, k_val)
            import sympy as sp

            return (retarded - advanced) * sp.coth(self.omega / (2 * self.temperature))

    def __getattr__(self, name: str) -> Any:
        """Delegate any missing attributes/methods to legacy calculator."""
        if hasattr(self._legacy_calc, name):
            return getattr(self._legacy_calc, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Global factory instance for convenience
_global_calculator_factory = CalculatorFactory()


def create_propagator_calculator(
    context: str | CalculatorContext, msrjd_action: MSRJDAction, **kwargs: Any
) -> PropagatorCalculatorBase:
    """Create propagator calculator using global factory."""
    return _global_calculator_factory.create_propagator_calculator(context, msrjd_action, **kwargs)


def get_calculator_factory() -> CalculatorFactory:
    """Get global calculator factory instance."""
    return _global_calculator_factory
