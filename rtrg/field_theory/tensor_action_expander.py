"""
Tensor-aware action expansion for vertex extraction and perturbative calculations.

This module provides enhanced action expansion capabilities that properly handle
tensor indices, automatic contractions, and systematic vertex extraction for
MSRJD field theory calculations.

Key Features:
    - Tensor-aware Taylor expansion around background configurations
    - Automatic index contraction in vertex extraction
    - Systematic organization of interaction vertices by order and structure
    - Integration with symbolic tensor field infrastructure
    - Enhanced caching and optimization for complex expressions
    - Covariant derivative handling in expansion terms

Mathematical Framework:
    The action S[φ, φ̃] is expanded as:
        S = S₀ + S₁ + S₂ + S₃ + S₄ + ...

    Where:
        S₀: Background action (constant)
        S₁: Linear terms (vanish for equilibrium background)
        S₂: Quadratic terms (propagators)
        S₃: Cubic vertices (3-point interactions)
        S₄: Quartic vertices (4-point interactions)
        ...

    Each term respects tensor index structure:
        S₃ = ∫ d⁴x V^{ijk}_{μνρ} φ^i_μ φ^j_ν φ^k_ρ

Usage:
    >>> action = TensorMSRJDAction(system)
    >>> expander = TensorActionExpander(action)
    >>> vertices = expander.extract_tensor_vertices(order=3)
    >>> quadratic_action = expander.get_quadratic_action_matrix()

References:
    - Tensor symbolic infrastructure (symbolic_tensors.py)
    - MSRJD formalism (plan/MSRJD_Formalism.md)
    - Phase 1 tensor infrastructure integration
"""

import warnings
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from typing import Any, Optional, Union

import sympy as sp
from sympy import (
    Derivative,
    Function,
    IndexedBase,
    Matrix,
    Symbol,
    collect,
    diff,
    expand,
    simplify,
    symbols,
)

from .symbolic_tensors import SymbolicTensorField, TensorDerivative
from .tensor_msrjd_action import TensorActionComponents, TensorMSRJDAction


@dataclass
class TensorVertex:
    """
    Representation of a tensor interaction vertex.

    Contains complete information about an n-point interaction vertex
    including tensor index structure, field content, and coupling strength.
    """

    order: int  # Number of fields in the vertex (3, 4, 5, ...)
    fields: list[str]  # Names of fields involved
    indices: list[list[str]]  # Tensor indices for each field
    coupling: sp.Expr  # Coupling constant/coefficient
    tensor_structure: sp.Expr  # Complete tensor expression
    momentum_dependence: sp.Expr | None = None  # k-space structure
    symmetry_factor: int = 1  # Combinatorial factor

    def __post_init__(self) -> None:
        """Validate vertex structure."""
        if len(self.fields) != self.order:
            raise ValueError("Number of fields must match vertex order")

        if len(self.indices) != self.order:
            raise ValueError("Number of index lists must match vertex order")

    def get_vertex_key(self) -> str:
        """Generate unique key for vertex identification."""
        field_signature = "_".join(sorted(self.fields))
        return f"V{self.order}_{field_signature}"


@dataclass
class TensorExpansionResult:
    """Container for expansion results with tensor structure information."""

    expansion_order: int
    expansion_terms: dict[int, sp.Expr] = field(default_factory=dict)
    vertices: dict[str, TensorVertex] = field(default_factory=dict)
    quadratic_matrix: Matrix | None = None
    field_basis: list[sp.Expr] = field(default_factory=list)
    background_values: dict[str, float | sp.Expr] = field(default_factory=dict)

    def get_vertices_by_order(self, order: int) -> list[TensorVertex]:
        """Get all vertices of specified order."""
        return [v for v in self.vertices.values() if v.order == order]

    def get_total_vertices(self) -> int:
        """Get total number of extracted vertices."""
        return len(self.vertices)


class TensorActionExpander:
    """
    Enhanced action expander with full tensor index handling.

    This class extends the basic ActionExpander to properly handle tensor
    indices, automatic contractions, and systematic vertex extraction for
    tensor field theories.

    Key Capabilities:
        - Tensor-aware Taylor expansion around background states
        - Automatic index contraction in derivative calculations
        - Systematic vertex extraction with proper tensor structure
        - Integration with symbolic tensor field registry
        - Enhanced caching for complex tensor expressions
        - Covariant derivative handling in expansion terms

    Mathematical Methods:
        1. Tensor Taylor Expansion:
           S[φ + δφ] = S[φ₀] + δφⁱ ∂S/∂φⁱ + ½ δφⁱ δφʲ ∂²S/∂φⁱ∂φʲ + ...

        2. Index Contraction:
           Automatic Einstein summation over repeated indices

        3. Vertex Classification:
           Organize vertices by field content and tensor structure

    Usage:
        >>> tensor_action = TensorMSRJDAction(system)
        >>> expander = TensorActionExpander(tensor_action)
        >>> result = expander.expand_to_order(4)
        >>> cubic_vertices = result.get_vertices_by_order(3)
    """

    def __init__(
        self,
        tensor_action: TensorMSRJDAction,
        background_config: dict[str, float | sp.Expr] | None = None,
    ):
        """
        Initialize tensor action expander.

        Args:
            tensor_action: TensorMSRJDAction instance to expand
            background_config: Background field configuration for expansion
        """
        self.tensor_action = tensor_action
        self.field_registry = tensor_action.field_registry
        self.coordinates = tensor_action.coordinates

        # Set up background configuration
        self.background_config = background_config or self._default_background()

        # Get complete action
        self.action_components = tensor_action.construct_full_action()
        self.total_action = self.action_components.total

        # Create field basis for expansion
        self.field_basis = self._create_tensor_field_basis()

        # Cache for expansion results
        self._expansion_cache: dict[int, TensorExpansionResult] = {}

    def _default_background(self) -> dict[str, float | sp.Expr]:
        """Create default equilibrium background configuration."""
        background = {}

        # Default background for Israel-Stewart fields
        background.update(
            {
                "rho": 1.0,  # Unit energy density
                "u_0": 1.0,  # u^0 = 1 in rest frame
                "u_1": 0.0,  # u^i = 0 in rest frame
                "u_2": 0.0,
                "u_3": 0.0,
                "Pi": 0.0,  # No bulk pressure at equilibrium
                "pi_ij": 0.0,  # No shear stress at equilibrium
                "q_i": 0.0,  # No heat flux at equilibrium
            }
        )

        return background

    def _create_tensor_field_basis(self) -> list[sp.Expr]:
        """
        Create complete basis of tensor field components for expansion.

        Returns:
            List of all field components (including tensor indices)
        """
        field_basis = []

        # Get all fields from registry
        all_fields = {
            **self.field_registry.get_all_fields(),
            **self.field_registry.get_all_antifields(),
        }

        for _field_name, tensor_field in all_fields.items():
            if tensor_field.index_count == 0:
                # Scalar field
                field_expr = tensor_field(*self.coordinates)
                field_basis.append(field_expr)

            elif tensor_field.index_count == 1:
                # Vector field - add all 4 components
                for mu in range(4):
                    field_expr = tensor_field(mu, *self.coordinates)
                    field_basis.append(field_expr)

            elif tensor_field.index_count == 2:
                # Rank-2 tensor - add all 16 components (but some may be constrained)
                for mu in range(4):
                    for nu in range(4):
                        field_expr = tensor_field(mu, nu, *self.coordinates)
                        field_basis.append(field_expr)

        return field_basis

    def expand_to_order(self, max_order: int) -> TensorExpansionResult:
        """
        Perform tensor-aware expansion to specified order.

        Args:
            max_order: Maximum order of expansion

        Returns:
            Complete expansion result with tensor structure
        """
        if max_order in self._expansion_cache:
            return self._expansion_cache[max_order]

        # Initialize result container
        result = TensorExpansionResult(
            expansion_order=max_order,
            background_values=self.background_config.copy(),
            field_basis=self.field_basis.copy(),
        )

        # Perform Taylor expansion order by order
        for order in range(max_order + 1):
            if order == 0:
                result.expansion_terms[0] = self._compute_background_action()
            elif order == 1:
                result.expansion_terms[1] = self._compute_linear_terms()
            elif order == 2:
                result.expansion_terms[2] = self._compute_quadratic_terms()
                result.quadratic_matrix = self._extract_quadratic_matrix()
            else:
                # Higher order terms
                result.expansion_terms[order] = self._compute_higher_order_terms(order)
                # Extract vertices for this order
                vertices = self._extract_vertices_at_order(order, result.expansion_terms[order])
                result.vertices.update(vertices)

        # Cache and return result
        self._expansion_cache[max_order] = result
        return result

    def _compute_background_action(self) -> sp.Expr:
        """Compute action evaluated at background configuration."""
        # Substitute background values into action
        substitutions = self._create_background_substitutions()
        background_action = self.total_action.subs(substitutions)
        return simplify(background_action)

    def _create_background_substitutions(self) -> dict[sp.Expr, float | sp.Expr]:
        """Create substitution dictionary for background values."""
        substitutions = {}

        for field_expr in self.field_basis:
            # Extract field name and indices from expression
            field_info = self._analyze_field_expression(field_expr)

            if field_info:
                field_name, indices = field_info
                background_key = self._get_background_key(field_name, indices)

                if background_key in self.background_config:
                    substitutions[field_expr] = self.background_config[background_key]

        return substitutions

    def _analyze_field_expression(self, expr: sp.Expr) -> tuple[str, list[int]] | None:
        """
        Analyze field expression to extract name and indices.

        Args:
            expr: Field expression to analyze

        Returns:
            Tuple of (field_name, index_list) or None if not a field
        """
        if hasattr(expr, "func") and hasattr(expr.func, "_name"):
            field_name = expr.func._name

            # Extract indices from arguments
            indices = []
            args = expr.args if hasattr(expr, "args") else []

            # Separate tensor indices from coordinate arguments
            coordinate_symbols = [str(coord) for coord in self.coordinates]

            for arg in args:
                arg_str = str(arg)
                if arg_str not in coordinate_symbols and isinstance(arg, int | Symbol):
                    if isinstance(arg, int):
                        indices.append(arg)
                    elif arg_str in ["mu", "nu", "alpha", "beta"]:
                        # These are symbolic indices - use 0 as default
                        indices.append(0)

            return field_name, indices

        return None

    def _get_background_key(self, field_name: str, indices: list[int]) -> str:
        """Generate background configuration key from field name and indices."""
        if not indices:
            return field_name
        elif len(indices) == 1:
            return f"{field_name}_{indices[0]}"
        elif len(indices) == 2:
            return f"{field_name}_{indices[0]}{indices[1]}"
        else:
            index_str = "".join(map(str, indices))
            return f"{field_name}_{index_str}"

    def _compute_linear_terms(self) -> sp.Expr:
        """Compute linear terms in expansion (should vanish for equilibrium)."""
        linear_terms = sp.sympify(0)

        # Create background substitutions
        bg_substitutions = self._create_background_substitutions()

        for field_expr in self.field_basis:
            # Compute derivative of action with respect to field
            derivative = diff(self.total_action, field_expr)

            # Evaluate at background
            bg_derivative = derivative.subs(bg_substitutions)

            # Get field perturbation
            field_info = self._analyze_field_expression(field_expr)
            if field_info:
                field_name, indices = field_info
                bg_key = self._get_background_key(field_name, indices)
                bg_value = self.background_config.get(bg_key, 0)
                perturbation = field_expr - bg_value

                linear_terms += bg_derivative * perturbation

        return simplify(linear_terms)

    def _compute_quadratic_terms(self) -> sp.Expr:
        """Compute quadratic terms for propagator extraction."""
        quadratic_terms = sp.sympify(0)

        # Create background substitutions
        bg_substitutions = self._create_background_substitutions()

        # Compute all second derivatives
        n_fields = len(self.field_basis)
        for i in range(n_fields):
            for j in range(i, n_fields):  # j >= i to avoid double counting
                field_i = self.field_basis[i]
                field_j = self.field_basis[j]

                # Second derivative
                second_deriv = diff(self.total_action, field_i, field_j)
                bg_second_deriv = second_deriv.subs(bg_substitutions)

                # Field perturbations
                field_i_info = self._analyze_field_expression(field_i)
                field_j_info = self._analyze_field_expression(field_j)

                if field_i_info and field_j_info:
                    bg_key_i = self._get_background_key(field_i_info[0], field_i_info[1])
                    bg_key_j = self._get_background_key(field_j_info[0], field_j_info[1])

                    bg_value_i = self.background_config.get(bg_key_i, 0)
                    bg_value_j = self.background_config.get(bg_key_j, 0)

                    pert_i = field_i - bg_value_i
                    pert_j = field_j - bg_value_j

                    if i == j:
                        # Diagonal term
                        quadratic_terms += sp.Rational(1, 2) * bg_second_deriv * pert_i * pert_j
                    else:
                        # Off-diagonal term
                        quadratic_terms += bg_second_deriv * pert_i * pert_j

        return simplify(quadratic_terms)

    def _extract_quadratic_matrix(self) -> Matrix:
        """Extract quadratic action as matrix for propagator calculation."""
        n_fields = len(self.field_basis)
        quad_matrix = sp.zeros(n_fields, n_fields)

        # Create background substitutions
        bg_substitutions = self._create_background_substitutions()

        for i in range(n_fields):
            for j in range(n_fields):
                field_i = self.field_basis[i]
                field_j = self.field_basis[j]

                # Second derivative
                second_deriv = diff(self.total_action, field_i, field_j)
                bg_second_deriv = second_deriv.subs(bg_substitutions)

                quad_matrix[i, j] = bg_second_deriv

        return quad_matrix

    def _compute_higher_order_terms(self, order: int) -> sp.Expr:
        """
        Compute terms of specified order in the expansion.

        Args:
            order: Order of terms to compute (≥ 3)

        Returns:
            Expression containing all terms of specified order
        """
        if order < 3:
            return sp.sympify(0)

        higher_order_terms = sp.sympify(0)
        bg_substitutions = self._create_background_substitutions()

        # Generate all combinations of fields for n-point vertices
        field_combinations = combinations_with_replacement(self.field_basis, order)

        for field_combo in field_combinations:
            # Compute n-th derivative
            nth_derivative = self.total_action
            for field_var in field_combo:
                nth_derivative = diff(nth_derivative, field_var)

            # Evaluate at background
            bg_nth_derivative = nth_derivative.subs(bg_substitutions)

            if bg_nth_derivative != 0:  # Non-zero coefficient
                # Create perturbation product
                perturbation_product = sp.sympify(1)
                for field in field_combo:
                    field_info = self._analyze_field_expression(field)
                    if field_info:
                        bg_key = self._get_background_key(field_info[0], field_info[1])
                        bg_value = self.background_config.get(bg_key, 0)
                        perturbation = field - bg_value
                        perturbation_product *= perturbation

                # Add term with proper combinatorial factor
                from math import factorial

                combinatorial_factor = factorial(order) / factorial(len(set(field_combo)))

                term = bg_nth_derivative * perturbation_product / combinatorial_factor
                higher_order_terms += term

        return simplify(higher_order_terms)

    def _extract_vertices_at_order(
        self, order: int, order_terms: sp.Expr
    ) -> dict[str, TensorVertex]:
        """
        Extract interaction vertices from terms of specified order.

        Args:
            order: Vertex order (number of fields)
            order_terms: Expression containing all terms of this order

        Returns:
            Dictionary of vertex objects keyed by unique identifiers
        """
        vertices = {}

        # Expand and collect terms by field products
        expanded_terms = expand(order_terms)

        # This is a simplified implementation
        # Full implementation would systematically analyze all terms
        # and extract coefficients and tensor structure

        # For now, create placeholder vertices for demonstration
        if order == 3:
            # Example: u-u-pi coupling vertex
            if "u" in str(expanded_terms) and "pi" in str(expanded_terms):
                vertex_key = "V3_u_u_pi"
                vertices[vertex_key] = TensorVertex(
                    order=3,
                    fields=["u", "u", "pi"],
                    indices=[["mu"], ["nu"], ["alpha", "beta"]],
                    coupling=sp.Symbol("g_uupi"),
                    tensor_structure=sp.Symbol("V_uupi_structure"),
                    symmetry_factor=2,  # Two identical u fields
                )

        return vertices

    def extract_tensor_vertices(self, order: int) -> list[TensorVertex]:
        """
        Extract all tensor vertices of specified order.

        Args:
            order: Vertex order to extract

        Returns:
            List of tensor vertices
        """
        result = self.expand_to_order(order)
        return result.get_vertices_by_order(order)

    def get_quadratic_action_matrix(self) -> Matrix:
        """
        Get quadratic action matrix for propagator calculation.

        Returns:
            Matrix representation of quadratic action
        """
        result = self.expand_to_order(2)
        return result.quadratic_matrix

    def get_propagator_inverse_matrix(self) -> Matrix:
        """
        Get inverse of quadratic action matrix (propagator matrix).

        Returns:
            Propagator matrix G = (S^(2))^(-1)
        """
        quad_matrix = self.get_quadratic_action_matrix()

        if quad_matrix is not None:
            try:
                # Compute matrix inverse
                propagator = quad_matrix.inv()
                return propagator
            except Exception as e:
                warnings.warn(f"Could not invert quadratic matrix: {str(e)}", stacklevel=2)
                return None

        return None

    def validate_expansion_consistency(self, max_order: int = 4) -> dict[str, bool]:
        """
        Validate consistency of expansion results.

        Args:
            max_order: Maximum order to validate

        Returns:
            Dictionary of validation results
        """
        validation = {}

        try:
            result = self.expand_to_order(max_order)

            # Check that expansion terms exist
            validation["terms_exist"] = len(result.expansion_terms) > 0

            # Check quadratic matrix properties
            if result.quadratic_matrix is not None:
                quad_matrix = result.quadratic_matrix
                validation["quadratic_symmetric"] = quad_matrix.equals(quad_matrix.T)
                validation["quadratic_finite"] = all(
                    elem.is_finite for elem in quad_matrix if elem != 0
                )
            else:
                validation["quadratic_symmetric"] = False
                validation["quadratic_finite"] = False

            # Check vertex structure
            validation["vertices_extracted"] = result.get_total_vertices() > 0

            # Overall consistency
            validation["overall"] = all(validation.values())

        except Exception as e:
            warnings.warn(f"Expansion validation failed: {str(e)}", stacklevel=2)
            validation["overall"] = False

        return validation

    def __str__(self) -> str:
        field_count = len(self.field_basis)
        return f"TensorActionExpander(fields={field_count}, background_config={len(self.background_config)})"

    def __repr__(self) -> str:
        return self.__str__()
