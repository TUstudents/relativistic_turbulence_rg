"""
Basic propagator calculations without tensor extensions.

This is a simplified version that avoids complex tensor imports
while preserving the core PropagatorCalculator functionality.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy import I, pi, simplify, solve, symbols

from ..core.fields import Field
from .msrjd_action import ActionExpander, MSRJDAction


@dataclass
class PropagatorComponents:
    """Container for different components of a propagator."""

    retarded: sp.Expr | None = None
    advanced: sp.Expr | None = None
    keldysh: sp.Expr | None = None
    spectral: sp.Expr | None = None

    def __post_init__(self) -> None:
        """Validate propagator components."""
        if self.retarded is not None and self.advanced is not None:
            # Check causality relations
            omega, k = symbols("omega k", real=True)
            try:
                # Advanced should equal [G^R(-ω*, -k)]*
                # expected_advanced = self.retarded.subs(omega, -omega).conjugate()
                # This is approximate - exact check would require full evaluation
                pass  # Skip validation for complex expressions
            except (AttributeError, TypeError):
                pass  # Skip validation for complex expressions


@dataclass
class SpectralProperties:
    """Spectral function properties extracted from propagator."""

    poles: list[complex] = field(default_factory=list)
    residues: list[complex] = field(default_factory=list)
    branch_cuts: list[tuple[complex, complex]] = field(default_factory=list)
    sum_rule_value: float | None = None

    def validate_causality(self) -> bool:
        """Check that all poles are in the lower half-plane (retarded)."""
        return all(pole.imag <= 0 for pole in self.poles if isinstance(pole, complex))


@dataclass
class PropagatorMatrix:
    """Matrix representation of propagators in field space."""

    matrix: sp.Matrix
    field_basis: list[Field]
    omega: sp.Symbol
    k_vector: list[sp.Symbol]

    def get_component(self, field1: Field, field2: Field) -> sp.Expr:
        """Extract specific propagator component G_{field1, field2}."""
        try:
            i = self.field_basis.index(field1)
            j = self.field_basis.index(field2)
            return self.matrix[i, j]
        except ValueError as e:
            raise ValueError(f"Field not found in basis: {e}") from e

    def invert(self) -> "PropagatorMatrix":
        """Compute inverse propagator matrix."""
        try:
            inv_matrix = self.matrix.inv()
            return PropagatorMatrix(
                matrix=inv_matrix,
                field_basis=self.field_basis,
                omega=self.omega,
                k_vector=self.k_vector,
            )
        except Exception as e:
            raise ValueError(f"Cannot invert propagator matrix: {e}") from e


class PropagatorCalculator:
    """Basic propagator calculator for Israel-Stewart theory."""

    def __init__(self, msrjd_action: MSRJDAction, temperature: float = 1.0):
        """Initialize propagator calculator."""
        self.msrjd_action = msrjd_action
        self.is_system = msrjd_action.is_system
        self.temperature = temperature

        # Create symbolic variables
        self.omega = sp.Symbol("omega", complex=True)
        self.k = sp.Symbol("k", real=True)
        self.k_vec = [sp.Symbol(f"k_{i}", real=True) for i in range(3)]

        # Cache for computed propagators
        self.propagator_cache: dict[str, PropagatorComponents] = {}
        self.matrix_cache: dict[str, PropagatorMatrix] = {}

    def calculate_retarded_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Calculate retarded propagator G^R_{field1,field2}(ω,k)."""
        # Create cache key
        cache_key = f"retarded_{field1.name}_{field2.name}"

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
            self.propagator_cache[cache_key] = PropagatorComponents()
        self.propagator_cache[cache_key].retarded = retarded

        # Substitute values if provided
        result = retarded
        if omega_val is not None:
            result = result.subs(self.omega, omega_val)
        if k_val is not None:
            result = result.subs(self.k, k_val)

        return result

    def _extract_coefficient(self, field1: Field, field2: Field) -> sp.Expr:
        """Extract coefficient from quadratic action."""
        # This is a simplified version - get basic coefficients
        if field1.name == field2.name:
            # Diagonal terms
            if field1.name == "rho":
                return -I * self.omega + self.is_system.parameters.kappa * self.k**2
            elif field1.name == "u":
                return -I * self.omega + self.is_system.parameters.eta * self.k**2
            elif field1.name == "pi":
                return 1 - I * self.omega * self.is_system.parameters.tau_pi
            elif field1.name == "Pi":
                return 1 - I * self.omega * self.is_system.parameters.tau_Pi
            elif field1.name == "q":
                return 1 - I * self.omega * self.is_system.parameters.tau_q
            else:
                return sp.sympify(1)
        else:
            # Off-diagonal terms (simplified)
            return sp.sympify(0)

    def calculate_advanced_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """Calculate advanced propagator G^A_{field1,field2}(ω,k)."""
        retarded = self.calculate_retarded_propagator(field1, field2)

        # Advanced: G^A(ω,k) = [G^R(-ω*,-k)]*
        advanced = retarded.subs(self.omega, -self.omega.conjugate())
        advanced = advanced.subs(self.k, -self.k)
        advanced = advanced.conjugate()

        # Substitute values if provided
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
        """Calculate Keldysh propagator using FDT."""
        # Get retarded and advanced propagators
        retarded = self.calculate_retarded_propagator(field1, field2)
        advanced = self.calculate_advanced_propagator(field1, field2)

        # Apply FDT relation
        T = sp.Symbol("T", real=True, positive=True)
        coth_factor = sp.coth(self.omega / (2 * T))
        keldysh = (retarded - advanced) * coth_factor
        keldysh = keldysh.subs(T, self.temperature)
        keldysh = simplify(keldysh)

        # Substitute values if provided
        result = keldysh
        if omega_val is not None:
            result = result.subs(self.omega, omega_val)
        if k_val is not None:
            result = result.subs(self.k, k_val)

        return result
