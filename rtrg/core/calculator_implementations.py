"""
Concrete implementations of unified calculator classes.

This module provides the concrete calculator implementations that extend
the abstract base classes, including SimplePropagatorCalculator and
EnhancedPropagatorCalculator with their specific calculation strategies.
"""

import warnings
from typing import Optional

import sympy as sp
from sympy import I, pi, simplify, solve

from ..field_theory.msrjd_action import MSRJDAction
from .calculators import PropagatorCalculatorBase
from .fields import Field


class SimplePropagatorCalculator(PropagatorCalculatorBase):
    """
    Simple propagator calculator for Israel-Stewart theory.

    Provides basic propagator calculations without complex tensor
    handling, suitable for initial analysis and testing.

    Migrated from: rtrg.field_theory.propagators_simple.PropagatorCalculator
    """

    def __init__(
        self, msrjd_action: MSRJDAction, temperature: float = 1.0, enable_caching: bool = True
    ):
        """
        Initialize simple propagator calculator.

        Args:
            msrjd_action: MSRJD action for propagator extraction
            temperature: System temperature for FDT relations
            enable_caching: Whether to enable result caching
        """
        super().__init__(
            msrjd_action, metric=None, temperature=temperature, enable_caching=enable_caching
        )

        # Store reference for compatibility
        self.msrjd_action = msrjd_action

    def calculate_retarded_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Calculate retarded propagator G^R_{field1,field2}(ω,k)."""
        # Create cache key
        cache_key = self._create_propagator_cache_key(
            "retarded", field1, field2, omega_val=omega_val, k_val=k_val
        )

        if cache_key in self.propagator_cache:
            cached = self.propagator_cache[cache_key]
            if cached.retarded is not None:
                result = cached.retarded
                # Substitute values if provided
                if omega_val is not None:
                    result = result.subs(self.omega, omega_val)
                if k_val is not None:
                    result = result.subs(self.k, k_val)
                return result

        # Extract coefficient from action
        coefficient = self._extract_coefficient(field1, field2)

        # Build retarded propagator: G^R = 1 / coefficient
        retarded = 1 / coefficient
        retarded = simplify(retarded)

        # Cache result
        if cache_key not in self.propagator_cache:
            from .calculators import PropagatorComponents

            self.propagator_cache[cache_key] = PropagatorComponents()
        self.propagator_cache[cache_key].retarded = retarded

        # Apply substitutions if provided
        result = retarded
        if omega_val is not None:
            result = result.subs(self.omega, omega_val)
        if k_val is not None:
            result = result.subs(self.k, k_val)

        return result

    def calculate_advanced_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Calculate advanced propagator G^A_{field1,field2}(ω,k)."""
        # Advanced propagator is complex conjugate of retarded with ω → -ω*
        retarded = self.calculate_retarded_propagator(field1, field2)

        # Apply advanced transformation: ω → -ω*
        advanced = retarded.subs(self.omega, -sp.conjugate(self.omega))
        advanced = sp.conjugate(advanced)
        advanced = simplify(advanced)

        # Apply substitutions if provided
        result = advanced
        if omega_val is not None:
            result = result.subs(self.omega, omega_val)
        if k_val is not None:
            result = result.subs(self.k, k_val)

        return result

    def calculate_keldysh_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Calculate Keldysh propagator G^K_{field1,field2}(ω,k) using FDT."""
        # Get retarded and advanced propagators
        retarded = self.calculate_retarded_propagator(field1, field2)
        advanced = self.calculate_advanced_propagator(field1, field2)

        # Fluctuation-dissipation theorem: G^K = (G^R - G^A) coth(ω/2T)
        keldysh = (retarded - advanced) * sp.coth(self.omega / (2 * self.temperature))
        keldysh = simplify(keldysh)

        # Apply substitutions if provided
        result = keldysh
        if omega_val is not None:
            result = result.subs(self.omega, omega_val)
        if k_val is not None:
            result = result.subs(self.k, k_val)

        return result

    def _extract_coefficient(self, field1: Field, field2: Field) -> sp.Expr:
        """
        Extract coefficient from MSRJD action for field pair.

        This is a simplified implementation that extracts coefficients
        from the quadratic part of the action.
        """
        try:
            # Get action components
            if hasattr(self.action, "construct_quadratic_action"):
                quadratic_action = self.action.construct_quadratic_action()
            else:
                # Fallback: try to extract from full action
                quadratic_action = getattr(self.action, "quadratic_terms", {})

            # Look for coefficient corresponding to field pair
            field_pair_key = f"{field1.name}_{field2.name}"
            reverse_key = f"{field2.name}_{field1.name}"

            coefficient = None

            # Try direct lookup
            if isinstance(quadratic_action, dict):
                coefficient = quadratic_action.get(field_pair_key) or quadratic_action.get(
                    reverse_key
                )

            # If no coefficient found, construct default based on field types
            if coefficient is None:
                coefficient = self._construct_default_coefficient(field1, field2)

            # Add small imaginary part for causality (retarded prescription)
            if coefficient.is_real or not coefficient.has(I):
                coefficient = coefficient + I * 1e-10

            return coefficient

        except Exception as e:
            warnings.warn(
                f"Failed to extract coefficient for {field1.name}-{field2.name}: {str(e)}"
            )
            # Return simple default coefficient
            return -I * self.omega + self.k**2

    def _construct_default_coefficient(self, field1: Field, field2: Field) -> sp.Expr:
        """
        Construct default coefficient based on field types.

        Provides reasonable defaults for common field combinations
        when coefficients cannot be extracted from action.
        """
        # Default structure: -iω + transport terms
        base_coefficient = -I * self.omega

        # Add momentum-dependent terms based on field types
        if field1.name == field2.name:  # Diagonal terms
            if field1.name in ["rho", "Pi"]:  # Scalar fields
                base_coefficient += self.k**2
            elif field1.name in ["u", "q"]:  # Vector fields
                base_coefficient += self.k**2
            elif field1.name == "pi":  # Tensor field
                base_coefficient += self.k**2
        else:  # Off-diagonal terms
            # Weaker coupling for off-diagonal elements
            base_coefficient += 0.1 * self.k**2

        return base_coefficient


class EnhancedPropagatorCalculator(PropagatorCalculatorBase):
    """
    Enhanced propagator calculator with advanced features.

    This will be implemented by refactoring the existing PropagatorCalculator
    from propagators.py to use the unified base class interface.

    Placeholder implementation - will be completed in next phase.
    """

    def calculate_retarded_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Enhanced retarded propagator calculation - placeholder."""
        # This will be implemented by migrating logic from propagators.py
        raise NotImplementedError("EnhancedPropagatorCalculator will be implemented in next phase")

    def calculate_advanced_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Enhanced advanced propagator calculation - placeholder."""
        raise NotImplementedError("EnhancedPropagatorCalculator will be implemented in next phase")

    def calculate_keldysh_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Enhanced Keldysh propagator calculation - placeholder."""
        raise NotImplementedError("EnhancedPropagatorCalculator will be implemented in next phase")
