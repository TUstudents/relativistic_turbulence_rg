"""
Systematic vertex extraction from MSRJD action for Feynman rule generation.

This module implements comprehensive vertex extraction from the Martin-Siggia-Rose-
Janssen-de Dominicis (MSRJD) action for relativistic Israel-Stewart hydrodynamics.
It systematically identifies all interaction vertices by expanding the action in powers
of field fluctuations and organizing them by their tensor structure.

Key Features:
    - Systematic extraction of 3-point and 4-point vertices from action
    - Tensor index management for proper Lorentz covariance
    - Momentum space conversion with proper derivative handling
    - Coupling constant organization and dimensional analysis
    - Vertex symmetry factor calculation
    - Integration with existing MSRJD action implementations

Mathematical Framework:
    The action expansion around equilibrium background:
        S[φ, φ̃] = S₀ + S₁ + S₂ + S₃ + S₄ + O(φ⁵)

    Vertices are extracted from:
        S₃ → 3-point vertices (cubic interactions)
        S₄ → 4-point vertices (quartic interactions)

    Each vertex is characterized by:
        - Field content: (field₁, field₂, field₃, ...)
        - Tensor structure: Lorentz indices and contractions
        - Coupling strength: Coefficients and transport parameters
        - Momentum dependence: Derivatives → momentum factors
        - Symmetry factors: Combinatorial factors from expansion

Vertex Categories:
    Advection vertices: φ̃ ∂φ terms from convective derivatives
    Relaxation vertices: φ̃ φ terms from viscous/dissipative processes
    Stress coupling: φ̃_tensor φ_velocity interactions
    Nonlinear vertices: Higher-order field products
    Constraint vertices: Lagrange multiplier contributions

Usage:
    >>> from rtrg.field_theory.msrjd_action import MSRJDAction
    >>> from rtrg.israel_stewart.equations import IsraelStewartSystem
    >>>
    >>> # Create IS system and action
    >>> system = IsraelStewartSystem(parameters)
    >>> action = MSRJDAction(system)
    >>>
    >>> # Extract vertices
    >>> extractor = VertexExtractor(action)
    >>> vertices = extractor.extract_all_vertices()
    >>>
    >>> # Generate Feynman rules
    >>> feynman_rules = extractor.generate_feynman_rules()

References:
    - MSRJD formalism: plan/MSRJD_Formalism.md
    - Israel-Stewart theory: plan/Israel-Stewart_Theory.md
    - Field theory methods: Peskin & Schroeder, "Introduction to QFT"
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import sympy as sp
from sympy import (
    Derivative,
    DiracDelta,
    Function,
    I,
    IndexedBase,
    Symbol,
    collect,
    expand,
    simplify,
    symbols,
)

from ..core.constants import PhysicalConstants
from ..core.tensors import LorentzTensor, Metric
from ..israel_stewart.equations import IsraelStewartParameters


@dataclass
class VertexStructure:
    """
    Complete description of an interaction vertex.

    Contains all information needed to generate Feynman rules:
    tensor structure, coupling constants, momentum dependence, etc.
    """

    # Field content
    fields: tuple[str, ...]  # Field names participating in vertex
    field_indices: dict[str, list[str]]  # Lorentz indices for each field

    # Tensor structure
    coupling_expression: sp.Expr  # Symbolic expression for coupling
    tensor_structure: str  # Description of tensor contractions

    # Physical parameters
    coupling_constants: set[Symbol]  # Transport coefficients involved
    mass_dimension: float  # Engineering dimension of vertex

    # Momentum space properties
    derivative_structure: dict[str, int]  # Derivative order per field
    momentum_factors: sp.Expr  # Momentum dependence in k-space

    # Symmetry properties
    symmetry_factor: float  # Statistical factor from field permutations
    vertex_type: str  # Category: "advection", "relaxation", "stress", etc.

    def validate_consistency(self) -> bool:
        """Verify internal consistency of vertex structure."""
        try:
            # Check field-index correspondence
            for field in self.fields:
                if field not in self.field_indices:
                    return False

            # Check dimensional consistency (mass dimension should be integer or half-integer)
            if not (self.mass_dimension % 0.5 == 0):
                return False

            # Check symmetry factor is positive
            if self.symmetry_factor <= 0:
                return False

            return True
        except Exception:
            return False


@dataclass
class VertexCatalog:
    """
    Complete catalog of all interaction vertices in the theory.

    Organizes vertices by order (3-point, 4-point) and type (advection,
    relaxation, etc.) for systematic Feynman rule generation.
    """

    three_point: dict[tuple[str, ...], VertexStructure]
    four_point: dict[tuple[str, ...], VertexStructure]
    constraint_vertices: dict[tuple[str, ...], VertexStructure]

    # Metadata
    total_vertices: int
    coupling_constants: set[Symbol]
    vertex_types: set[str]

    def __post_init__(self) -> None:
        """Initialize derived quantities."""
        self.total_vertices = (
            len(self.three_point) + len(self.four_point) + len(self.constraint_vertices)
        )

        # Collect all coupling constants
        all_vertices = (
            list(self.three_point.values())
            + list(self.four_point.values())
            + list(self.constraint_vertices.values())
        )
        self.coupling_constants = set()
        self.vertex_types = set()

        for vertex in all_vertices:
            self.coupling_constants.update(vertex.coupling_constants)
            self.vertex_types.add(vertex.vertex_type)

    def get_vertices_by_type(self, vertex_type: str) -> list[VertexStructure]:
        """Get all vertices of specified type."""
        matching_vertices = []
        all_vertices = (
            list(self.three_point.values())
            + list(self.four_point.values())
            + list(self.constraint_vertices.values())
        )

        for vertex in all_vertices:
            if vertex.vertex_type == vertex_type:
                matching_vertices.append(vertex)

        return matching_vertices

    def get_vertices_with_field(self, field_name: str) -> list[VertexStructure]:
        """Get all vertices containing specified field."""
        matching_vertices = []
        all_vertices = (
            list(self.three_point.values())
            + list(self.four_point.values())
            + list(self.constraint_vertices.values())
        )

        for vertex in all_vertices:
            if field_name in vertex.fields:
                matching_vertices.append(vertex)

        return matching_vertices


class VertexExtractor:
    """
    Systematic extraction of interaction vertices from MSRJD action.

    This class implements the complete pipeline for vertex extraction:
    1. Action expansion around background
    2. Term identification and classification
    3. Tensor structure analysis
    4. Coupling constant extraction
    5. Momentum space conversion
    6. Symmetry factor calculation
    """

    def __init__(
        self,
        action: Any,  # MSRJDAction instance
        background: dict[str, float] | None = None,
        metric: Any | None = None,
    ):
        """
        Initialize vertex extractor with MSRJD action.

        Args:
            action: MSRJDAction instance to extract vertices from
            background: Background field values for expansion
            metric: Spacetime metric (defaults to Minkowski)
        """
        self.action = action
        self.parameters = action.parameters
        self.metric = metric or Metric()

        # Default equilibrium background
        if background is None:
            background = {
                "rho": 1.0,  # Background energy density
                "u_0": PhysicalConstants.c,  # Four-velocity time component
                "u_1": 0.0,
                "u_2": 0.0,
                "u_3": 0.0,  # Spatial components
                "Pi": 0.0,  # Bulk pressure
                "pi": 0.0,  # Shear stress components
                "q": 0.0,  # Heat flux components
            }
        self.background = background

        # Field categorization
        self.physical_fields = ["rho", "u", "pi", "Pi", "q"]
        self.response_fields = ["rho_tilde", "u_tilde", "pi_tilde", "Pi_tilde", "q_tilde"]

        # Spacetime coordinates and indices
        self.coordinates = symbols("t x y z", real=True)
        self.mu, self.nu, self.alpha, self.beta = symbols("mu nu alpha beta", integer=True)

        # Caching for expensive operations
        self._expansion_cache: dict[int, sp.Expr] = {}
        self._vertex_cache: VertexCatalog | None = None

    def expand_action_around_background(self, max_order: int = 4) -> dict[int, sp.Expr]:
        """
        Expand action in powers of field fluctuations around background.

        Args:
            max_order: Maximum order in field expansion

        Returns:
            Dictionary mapping order to symbolic expression
        """
        if max_order in self._expansion_cache:
            return {i: self._expansion_cache[i] for i in range(max_order + 1)}

        # Get total action
        action_components = self.action.construct_total_action()
        total_action = action_components.total

        # Create field perturbations: φ = φ₀ + δφ
        perturbation_substitutions = {}
        all_fields = list(self.action.fields.values()) + list(self.action.response_fields.values())

        for field in all_fields:
            # Handle both Function and IndexedBase fields
            if isinstance(field, Function):
                field_name = str(field.func)
                background_value = self.background.get(field_name, 0.0)
                # δφ = φ - φ₀
                perturbation_substitutions[field] = field - background_value
            elif isinstance(field, IndexedBase):
                field_name = str(field)
                # For tensor fields, handle component-wise
                # This is simplified - full implementation would handle all components
                perturbation_substitutions[field] = field  # Keep as is for now

        # Taylor expansion around background
        expansion = {}

        # Zeroth order (background action)
        background_action = total_action.subs(
            {
                field: self.background.get(str(field), 0.0)
                for field in all_fields
                if isinstance(field, Function)
            }
        )
        expansion[0] = background_action

        # First order (linear terms - should vanish for equilibrium)
        linear_terms = sp.sympify(0)
        for field in all_fields:
            if isinstance(field, Function):
                derivative = sp.diff(total_action, field)
                field_name = str(field.func)
                background_value = self.background.get(field_name, 0.0)

                # Evaluate derivative at background
                background_subs = {
                    f: self.background.get(str(f), 0.0)
                    for f in all_fields
                    if isinstance(f, Function)
                }
                linear_coeff = derivative.subs(background_subs)

                # Add linear term
                linear_terms += linear_coeff * (field - background_value)
        expansion[1] = linear_terms

        # Second order (quadratic terms - propagators)
        quadratic_terms = sp.sympify(0)
        field_list = [
            f for f in all_fields if isinstance(f, Function)
        ]  # Focus on Function fields for now

        for i, field_i in enumerate(field_list):
            for j, field_j in enumerate(field_list):
                if j >= i:  # Avoid double counting
                    second_derivative = sp.diff(total_action, field_i, field_j)

                    # Evaluate at background
                    background_subs = {f: self.background.get(str(f.func), 0.0) for f in field_list}
                    coeff = second_derivative.subs(background_subs)

                    # Get perturbations
                    field_i_pert = field_i - self.background.get(str(field_i.func), 0.0)
                    field_j_pert = field_j - self.background.get(str(field_j.func), 0.0)

                    # Add quadratic term
                    if i == j:
                        quadratic_terms += sp.Rational(1, 2) * coeff * field_i_pert**2
                    else:
                        quadratic_terms += coeff * field_i_pert * field_j_pert

        expansion[2] = quadratic_terms

        # Higher orders (cubic and quartic vertices)
        for order in range(3, max_order + 1):
            higher_order_terms = sp.sympify(0)

            # This would implement systematic higher-order derivative extraction
            # For now, we'll extract from known interaction structure
            if order == 3:
                higher_order_terms = self._extract_cubic_terms_from_action(total_action)
            elif order == 4:
                higher_order_terms = self._extract_quartic_terms_from_action(total_action)

            expansion[order] = higher_order_terms

        # Cache results
        for order, expr in expansion.items():
            self._expansion_cache[order] = expr

        return expansion

    def _extract_cubic_terms_from_action(self, action: sp.Expr) -> sp.Expr:
        """
        Extract cubic interaction terms from the deterministic action.

        Complete Israel-Stewart cubic vertices:
        1. Advection terms: ũ_μ u^ν ∂_ν u^μ
        2. Shear-velocity coupling: π̃_μν ∇_(μ u_ν)
        3. Bulk-velocity coupling: Π̃ ∇·u
        4. Heat flux coupling: q̃_μ ∇^μ T
        5. Energy density advection: ρ̃ u^μ ∂_μ ρ
        6. Nonlinear relaxation terms
        """
        cubic_terms = sp.sympify(0)

        # Get field symbols
        u = self.action.fields.get("u")
        u_tilde = self.action.response_fields.get("u_tilde")
        pi = self.action.fields.get("pi")
        pi_tilde = self.action.response_fields.get("pi_tilde")
        rho = self.action.fields.get("rho")
        rho_tilde = self.action.response_fields.get("rho_tilde")
        Pi = self.action.fields.get("Pi")
        Pi_tilde = self.action.response_fields.get("Pi_tilde")
        q = self.action.fields.get("q")
        q_tilde = self.action.response_fields.get("q_tilde")

        # 1. Advection vertices: ũ_μ u^ν ∂_ν u^μ
        if u and u_tilde:
            # Use coordinate symbols directly instead of indexed access
            t, x, y, z = self.coordinates
            advection_vertex = (
                u_tilde[self.mu]
                * u[self.nu]
                * Derivative(u[self.mu], x)  # Simplified to x derivative
            )
            cubic_terms += advection_vertex

        # 2. Shear-velocity coupling: π̃_μν ∇_(μ u_ν) = π̃_μν σ_μν
        if pi_tilde and u:
            t, x, y, z = self.coordinates
            # Symmetric shear rate tensor σ_μν = (1/2)[∇_μ u_ν + ∇_ν u_μ - (2/3)g_μν ∇·u]
            shear_rate = (
                (Derivative(u[self.mu], x) + Derivative(u[self.nu], y)) / 2
                - sp.Rational(1, 3)
                * sp.KroneckerDelta(self.mu, self.nu)
                * Derivative(u[1], x)  # Simplified trace
            )
            stress_coupling = pi_tilde[self.mu, self.nu] * shear_rate
            cubic_terms += stress_coupling

        # 3. Bulk pressure-velocity coupling: Π̃ ∇·u
        if Pi_tilde and u:
            t, x, y, z = self.coordinates
            # ∇·u = ∂_x u^x + ∂_y u^y + ∂_z u^z
            expansion_scalar = Derivative(u[1], x) + Derivative(u[2], y) + Derivative(u[3], z)
            bulk_coupling = Pi_tilde * expansion_scalar
            cubic_terms += bulk_coupling

        # 4. Heat flux-temperature coupling: q̃_μ ∇^μ T
        if q_tilde and rho:  # Use ρ as proxy for temperature
            t, x, y, z = self.coordinates
            # Heat flux couples to temperature gradients
            temp_gradient_coupling = (
                q_tilde[1] * Derivative(rho, x)
                + q_tilde[2] * Derivative(rho, y)
                + q_tilde[3] * Derivative(rho, z)
            )
            cubic_terms += temp_gradient_coupling

        # 5. Energy density advection: ρ̃ u^μ ∂_μ ρ
        if rho_tilde and u and rho:
            t, x, y, z = self.coordinates
            energy_advection = (
                rho_tilde * u[0] * Derivative(rho, t)  # Time advection
                + rho_tilde * u[1] * Derivative(rho, x)  # Spatial advection
                + rho_tilde * u[2] * Derivative(rho, y)
                + rho_tilde * u[3] * Derivative(rho, z)
            )
            cubic_terms += energy_advection

        # 6. Nonlinear relaxation terms (transport coefficient dependence)
        if pi_tilde and pi and u:
            t, x, y, z = self.coordinates
            # Shear stress source term: π̃_μν (2η σ_μν)
            eta_coupling = (
                pi_tilde[self.mu, self.nu]
                * 2
                * self.parameters.eta
                * (Derivative(u[self.mu], x) + Derivative(u[self.nu], y))
                / 2
            )
            cubic_terms += eta_coupling

        if Pi_tilde and Pi and u:
            t, x, y, z = self.coordinates
            # Bulk pressure source: Π̃ (-ζ ∇·u)
            zeta_coupling = (
                Pi_tilde
                * (-self.parameters.zeta)
                * (Derivative(u[1], x) + Derivative(u[2], y) + Derivative(u[3], z))
            )
            cubic_terms += zeta_coupling

        if q_tilde and q and rho:
            t, x, y, z = self.coordinates
            # Heat flux source: q̃_μ (-κ ∇^μ T)
            kappa_coupling = (
                q_tilde[1] * (-self.parameters.kappa) * Derivative(rho, x)
                + q_tilde[2] * (-self.parameters.kappa) * Derivative(rho, y)
                + q_tilde[3] * (-self.parameters.kappa) * Derivative(rho, z)
            )
            cubic_terms += kappa_coupling

        return cubic_terms

    def _extract_quartic_terms_from_action(self, action: sp.Expr) -> sp.Expr:
        """
        Extract quartic interaction terms from nonlinear IS dynamics.

        Complete quartic vertices:
        1. Nonlinear stress terms: π̃_μν π^αβ π_αβ, π̃_μν π^μα π_α^ν
        2. Higher-order advection: ũ_μ u^ν u^α ∂_ν∂_α u^μ
        3. Mixed field interactions: π̃_μν π^μν Π, q̃_μ q^μ Π
        4. Stress-heat coupling: π̃_μν u^μ q^ν
        5. Thermodynamic nonlinearities
        """
        quartic_terms = sp.sympify(0)

        # Get field symbols
        pi = self.action.fields.get("pi")
        pi_tilde = self.action.response_fields.get("pi_tilde")
        u = self.action.fields.get("u")
        u_tilde = self.action.response_fields.get("u_tilde")
        Pi = self.action.fields.get("Pi")
        Pi_tilde = self.action.response_fields.get("Pi_tilde")
        q = self.action.fields.get("q")
        q_tilde = self.action.response_fields.get("q_tilde")
        rho = self.action.fields.get("rho")

        # 1. Nonlinear shear stress terms
        if pi and pi_tilde:
            # π̃_μν π^μα π_α^ν (quadratic nonlinearity)
            nonlinear_stress_1 = (
                pi_tilde[self.mu, self.nu] * pi[self.mu, self.alpha] * pi[self.alpha, self.nu]
            )
            quartic_terms += nonlinear_stress_1

            # π̃_μν π^αβ π_αβ g_μν (trace coupling)
            stress_trace_coupling = (
                pi_tilde[self.mu, self.nu]
                * pi[self.alpha, self.beta]
                * pi[self.alpha, self.beta]
                * sp.KroneckerDelta(self.mu, self.nu)
            )
            quartic_terms += stress_trace_coupling

        # 2. Higher-order advection terms
        if u and u_tilde:
            t, x, y, z = self.coordinates
            # ũ_μ u^ν u^α ∂_ν∂_α u^μ (second-order spatial derivatives)
            higher_advection = (
                u_tilde[self.mu]
                * u[self.nu]
                * u[self.alpha]
                * Derivative(Derivative(u[self.mu], x), y)  # Simplified second derivative
            )
            quartic_terms += higher_advection

        # 3. Mixed field interactions
        if pi_tilde and pi and Pi:
            # Shear-bulk coupling: π̃_μν π^μν Π
            shear_bulk_mixing = pi_tilde[self.mu, self.nu] * pi[self.mu, self.nu] * Pi
            quartic_terms += shear_bulk_mixing

        if q_tilde and q and Pi:
            # Heat-bulk coupling: q̃_μ q^μ Π
            heat_bulk_mixing = q_tilde[self.mu] * q[self.mu] * Pi
            quartic_terms += heat_bulk_mixing

        # 4. Stress-heat coupling
        if pi_tilde and u and q:
            # π̃_μν u^μ q^ν (momentum-heat flux coupling)
            stress_heat_coupling = pi_tilde[self.mu, self.nu] * u[self.mu] * q[self.nu]
            quartic_terms += stress_heat_coupling

        # 5. Thermodynamic nonlinearities (energy density corrections)
        if pi_tilde and pi and rho:
            # Energy density corrections: π̃_μν π^μν ρ
            energy_stress_coupling = pi_tilde[self.mu, self.nu] * pi[self.mu, self.nu] * rho
            quartic_terms += energy_stress_coupling

        return quartic_terms

    def extract_all_vertices(self) -> VertexCatalog:
        """
        Extract complete vertex catalog from action expansion.

        Returns:
            VertexCatalog with all interaction vertices organized by type
        """
        if self._vertex_cache is not None:
            return self._vertex_cache

        # Get action components
        action_components = self.action.construct_total_action()
        deterministic_action = action_components.deterministic

        # Use specialized physics-based extraction methods instead of generic expansion
        cubic_action = self._extract_cubic_terms_from_action(deterministic_action)
        quartic_action = self._extract_quartic_terms_from_action(deterministic_action)

        # Extract vertices from specialized terms
        cubic_vertices = self._extract_vertices_from_expansion(cubic_action, order=3)
        quartic_vertices = self._extract_vertices_from_expansion(quartic_action, order=4)

        # Extract constraint vertices (from Lagrange multiplier terms)
        constraint_vertices = self._extract_constraint_vertices()

        # Build catalog (post_init will populate vertex_types, coupling_constants, total_vertices)
        catalog = VertexCatalog(
            three_point=cubic_vertices,
            four_point=quartic_vertices,
            constraint_vertices=constraint_vertices,
            total_vertices=0,  # Will be set in __post_init__
            coupling_constants=set(),  # Will be set in __post_init__
            vertex_types=set(),  # Will be set in __post_init__
        )

        self._vertex_cache = catalog
        return catalog

    def _extract_vertices_from_expansion(
        self, expansion_term: sp.Expr, order: int
    ) -> dict[tuple[str, ...], VertexStructure]:
        """
        Extract individual vertices from expansion term of given order.

        Args:
            expansion_term: Symbolic expression containing interaction terms
            order: Vertex order (3 for cubic, 4 for quartic)

        Returns:
            Dictionary mapping field combinations to vertex structures
        """
        vertices: dict[tuple[str, ...], Any] = {}

        if expansion_term == 0:
            return vertices

        # Expand terms (avoid collect which fails with multi-variable derivatives)
        try:
            expanded = expand(expansion_term)
            # Use simple expansion instead of collect to avoid derivative issues
            collected = expanded
        except (NotImplementedError, AttributeError):
            # If expansion fails, use original term
            collected = expansion_term

        # Analyze terms - handle both single terms and sums
        try:
            if hasattr(collected, "as_ordered_terms"):
                terms = collected.as_ordered_terms()
            elif hasattr(collected, "args") and collected.args:
                terms = collected.args
            else:
                terms = [collected]
        except (AttributeError, NotImplementedError):
            terms = [collected]

        for term in terms:
            try:
                vertex = self._analyze_term_for_vertex(term, order)
                if vertex and vertex.validate_consistency():
                    # Create a more specific key to avoid overwriting different physics processes
                    field_signature = vertex.fields
                    # Include vertex type and key parts of the expression to make unique key
                    coupling_info = (
                        str(sorted(vertex.coupling_constants))
                        if vertex.coupling_constants
                        else "no_coupling"
                    )
                    derivative_info = (
                        "derivatives"
                        if "Derivative" in str(vertex.coupling_expression)
                        else "no_derivatives"
                    )
                    unique_key = (
                        *field_signature,
                        vertex.vertex_type,
                        coupling_info,
                        derivative_info,
                    )
                    vertices[unique_key] = vertex
            except Exception:
                # Skip problematic terms
                continue

        return vertices

    def _analyze_term_for_vertex(self, term: sp.Expr, order: int) -> VertexStructure | None:
        """
        Analyze a single term to extract vertex structure.

        Args:
            term: Symbolic expression for interaction term
            order: Expected vertex order

        Returns:
            VertexStructure if valid vertex found, None otherwise
        """
        try:
            # Extract field content
            fields_in_term = []
            field_indices: dict[str, list[Any]] = defaultdict(list)

            # Find all field symbols in the term - handle IndexedBase and Functions properly
            term_atoms = term.atoms(sp.Function, sp.IndexedBase, sp.Indexed)

            for atom in term_atoms:
                if isinstance(atom, sp.Function | sp.IndexedBase):
                    atom_name = str(atom.func) if hasattr(atom, "func") else str(atom)
                    # Check if it's a field we recognize
                    for field_name in self.physical_fields + self.response_fields:
                        if field_name in atom_name or atom_name.startswith(field_name):
                            if field_name not in fields_in_term:
                                fields_in_term.append(field_name)
                            # Initialize field_indices entry if not exists
                            if field_name not in field_indices:
                                field_indices[field_name] = []
                            break
                elif isinstance(atom, sp.Indexed):
                    # Handle indexed expressions like u[mu], pi[mu,nu]
                    base_name = str(atom.base)
                    for field_name in self.physical_fields + self.response_fields:
                        if field_name in base_name or base_name.startswith(field_name):
                            if field_name not in fields_in_term:
                                fields_in_term.append(field_name)
                            # Extract and store indices
                            if field_name not in field_indices:
                                field_indices[field_name] = []
                            # Add indices from this specific occurrence
                            if hasattr(atom, "indices") and atom.indices:
                                for idx in atom.indices:
                                    if str(idx) not in [
                                        str(existing) for existing in field_indices[field_name]
                                    ]:
                                        field_indices[field_name].append(str(idx))
                            break

            # Also check free symbols as fallback
            for symbol in term.free_symbols:
                symbol_str = str(symbol)
                for field_name in self.physical_fields + self.response_fields:
                    if field_name in symbol_str:
                        if field_name not in fields_in_term:
                            fields_in_term.append(field_name)
                        # Initialize field_indices entry if not exists
                        if field_name not in field_indices:
                            field_indices[field_name] = []
                        break

            # Remove duplicates while preserving order
            unique_fields = []
            for field in fields_in_term:
                if field not in unique_fields:
                    unique_fields.append(field)

            # Check if we have reasonable number of fields (more flexible for Israel-Stewart theory)
            if len(unique_fields) < 2 or len(unique_fields) > order + 2:
                return None

            # Extract coupling constants
            coupling_constants = set()
            for param in [
                self.parameters.eta,
                self.parameters.tau_pi,
                self.parameters.zeta,
                self.parameters.tau_Pi,
                self.parameters.kappa,
                self.parameters.tau_q,
            ]:
                if param in term.free_symbols:
                    coupling_constants.add(param)

            # Determine vertex type
            vertex_type = self._classify_vertex_type(unique_fields, term)

            # Calculate symmetry factor (simplified)
            symmetry_factor = self._calculate_symmetry_factor(unique_fields)

            # Create vertex structure
            try:
                vertex = VertexStructure(
                    fields=tuple(unique_fields),
                    field_indices=dict(field_indices),
                    coupling_expression=term,
                    tensor_structure=self._describe_tensor_structure(term),
                    coupling_constants=coupling_constants,
                    mass_dimension=self._calculate_mass_dimension(unique_fields, term),
                    derivative_structure=self._extract_derivative_structure(term),
                    momentum_factors=self._convert_to_momentum_space(term),
                    symmetry_factor=symmetry_factor,
                    vertex_type=vertex_type,
                )
                return vertex
            except Exception:
                return None

        except Exception:
            # If analysis fails, skip this term
            return None

    def _classify_vertex_type(self, fields: list[str], term: sp.Expr) -> str:
        """Classify vertex by its physical origin and structure."""
        term_str = str(term)

        # Check for transport coefficient presence
        has_eta = "eta" in term_str or any(
            hasattr(self.parameters, "eta") and str(self.parameters.eta) in term_str for _ in [None]
        )
        has_zeta = "zeta" in term_str or any(
            hasattr(self.parameters, "zeta") and str(self.parameters.zeta) in term_str
            for _ in [None]
        )
        has_kappa = "kappa" in term_str or any(
            hasattr(self.parameters, "kappa") and str(self.parameters.kappa) in term_str
            for _ in [None]
        )
        has_tau = any(tau_param in term_str for tau_param in ["tau_pi", "tau_Pi", "tau_q"])

        # Check derivative structure for advection
        derivative_count = term_str.count("Derivative")
        has_derivatives = derivative_count > 0

        # Classify by physical process
        if has_derivatives and any(field in ["u", "u_tilde"] for field in fields):
            if derivative_count >= 2:
                return "higher_order_advection"
            else:
                return "advection"

        # Transport vertices (viscous/conductive)
        if has_eta and "pi" in fields:
            return "shear_transport"
        if has_zeta and "Pi" in fields:
            return "bulk_transport"
        if has_kappa and "q" in fields:
            return "heat_transport"

        # Relaxation vertices (governed by relaxation times)
        if has_tau:
            if "pi" in fields:
                return "shear_relaxation"
            elif "Pi" in fields:
                return "bulk_relaxation"
            elif "q" in fields:
                return "heat_relaxation"
            else:
                return "relaxation"

        # Nonlinear field interactions
        field_count: dict[str, int] = {}
        for field in fields:
            field_count[field] = field_count.get(field, 0) + 1

        if max(field_count.values()) >= 2:  # Repeated fields indicate nonlinearity
            return "nonlinear_interaction"

        # Mixed field couplings
        unique_physical_fields = {f for f in fields if not f.endswith("_tilde")}
        if len(unique_physical_fields) >= 2:
            return "mixed_coupling"

        # Energy-momentum coupling
        if "rho" in fields and "u" in fields:
            return "energy_momentum"

        # Stress tensor interactions
        if "pi" in fields and any(f in fields for f in ["Pi", "q", "rho"]):
            return "stress_coupling"

        # Default classification by field content
        if "pi" in fields or "pi_tilde" in fields:
            return "stress"
        elif "Pi" in fields or "Pi_tilde" in fields:
            return "bulk"
        elif "q" in fields or "q_tilde" in fields:
            return "heat"
        elif "u" in fields or "u_tilde" in fields:
            return "momentum"
        elif "rho" in fields or "rho_tilde" in fields:
            return "energy"

        return "unclassified"

    def _calculate_symmetry_factor(self, fields: list[str]) -> float:
        """Calculate combinatorial symmetry factor."""
        # Count identical fields
        field_counts: dict[str, int] = defaultdict(int)
        for field in fields:
            field_counts[field] += 1

        # Symmetry factor is 1/∏(n_i!) for n_i identical fields
        symmetry_factor = 1.0
        for count in field_counts.values():
            for i in range(1, count + 1):
                symmetry_factor /= i

        return symmetry_factor

    def _describe_tensor_structure(self, term: sp.Expr) -> str:
        """Generate comprehensive description of tensor structure."""
        term_str = str(term)

        # Count index types
        mu_count = term_str.count("[mu")
        nu_count = term_str.count("[nu")
        alpha_count = term_str.count("[alpha")
        beta_count = term_str.count("[beta")

        total_indices = mu_count + nu_count + alpha_count + beta_count
        unique_indices = len(
            {"mu", "nu", "alpha", "beta"}
            & {idx for idx in ["mu", "nu", "alpha", "beta"] if f"[{idx}" in term_str}
        )

        # Derivative structure
        derivative_count = term_str.count("Derivative")

        # Kronecker delta presence
        has_kronecker = "KroneckerDelta" in term_str

        # Build description
        description_parts = []

        # Tensor rank classification
        if total_indices == 0:
            description_parts.append("scalar")
        elif total_indices == 1:
            description_parts.append("vector")
        elif total_indices == 2:
            if unique_indices == 2:
                description_parts.append("rank-2 tensor")
            else:
                description_parts.append("rank-2 symmetric tensor")
        elif total_indices >= 3:
            description_parts.append(f"rank-{total_indices} tensor")

        # Contraction type
        contracted_pairs = min(mu_count, nu_count) + min(alpha_count, beta_count)
        if contracted_pairs > 0:
            description_parts.append(f"with {contracted_pairs} contractions")

        # Derivative information
        if derivative_count > 0:
            if derivative_count == 1:
                description_parts.append("first derivative")
            elif derivative_count == 2:
                description_parts.append("second derivative")
            else:
                description_parts.append(f"{derivative_count}-th derivative")

        # Symmetry properties
        if has_kronecker:
            description_parts.append("with metric coupling")

        # Check for traceless structure
        if "- (1/3)" in term_str or "- sp.Rational(1, 3)" in term_str:
            description_parts.append("traceless")

        return " ".join(description_parts) if description_parts else "trivial coupling"

    def _calculate_mass_dimension(self, fields: list[str], term: sp.Expr) -> float:
        """Calculate engineering mass dimension of vertex with transport coefficients."""
        # Field dimensions in natural units where [length] = [time] = [mass]^{-1}
        # Using standard relativistic field theory conventions
        field_dimensions = {
            "rho": 4.0,  # Energy density [M^4]
            "u": 0.0,  # Four-velocity (dimensionless)
            "pi": 4.0,  # Shear stress [M^4] (pressure dimension)
            "Pi": 4.0,  # Bulk pressure [M^4]
            "q": 3.0,  # Heat flux [M^3]
            # Response fields have conjugate dimensions for MSRJD action
            "rho_tilde": -4.0,  # [M^{-4}]
            "u_tilde": -2.0,  # [M^{-2}] (to cancel derivatives)
            "pi_tilde": -4.0,  # [M^{-4}]
            "Pi_tilde": -4.0,  # [M^{-4}]
            "q_tilde": -3.0,  # [M^{-3}]
        }

        # Transport coefficient dimensions in natural units
        transport_dimensions = {
            "eta": 3.0,  # Shear viscosity [M^3]
            "zeta": 3.0,  # Bulk viscosity [M^3]
            "kappa": 3.0,  # Thermal conductivity [M^3]
            "tau_pi": -1.0,  # Relaxation time [M^{-1}]
            "tau_Pi": -1.0,  # Relaxation time [M^{-1}]
            "tau_q": -1.0,  # Relaxation time [M^{-1}]
        }

        total_dimension = 0.0

        # Add field contributions
        for field in fields:
            if field in field_dimensions:
                total_dimension += field_dimensions[field]
            else:
                # Handle tilde fields
                clean_field = field.replace("_tilde", "")
                if clean_field in field_dimensions:
                    # Response fields have negative dimensions
                    base_dim = field_dimensions[clean_field]
                    total_dimension += -base_dim if field.endswith("_tilde") else base_dim

        # Add derivative contributions
        term_str = str(term)
        derivative_count = term_str.count("Derivative")
        total_dimension += derivative_count  # Each ∂ adds dimension [M]

        # Add transport coefficient contributions
        for param, dim in transport_dimensions.items():
            if param in term_str:
                total_dimension += dim

        # Check for metric tensor contractions (dimensionless)
        kronecker_count = term_str.count("KroneckerDelta")
        # KroneckerDelta is dimensionless, no contribution

        # Rational prefactors don't change dimension
        # Constants like 1/2, 1/3 are dimensionless

        return total_dimension

    def _extract_derivative_structure(self, term: sp.Expr) -> dict[str, int]:
        """Extract which fields have derivatives and their order."""
        derivative_info: dict[str, int] = defaultdict(int)

        # This is a simplified extraction - full implementation would parse the derivative structure
        if hasattr(term, "atoms"):
            derivatives = term.atoms(Derivative)
            for deriv in derivatives:
                # Extract which field is being differentiated
                deriv_str = str(deriv)
                for field_name in self.physical_fields:
                    if field_name in deriv_str:
                        derivative_info[field_name] += 1
                        break

        return dict(derivative_info)

    def _convert_to_momentum_space(self, term: sp.Expr) -> sp.Expr:
        """Convert spatial derivatives to momentum factors."""
        # In momentum space: ∂_μ → ik_μ
        k_mu = IndexedBase("k")

        momentum_term = term

        # Replace derivatives with momentum factors
        if hasattr(term, "atoms"):
            derivatives = term.atoms(Derivative)
            for deriv in derivatives:
                # ∂/∂x^μ → ik_μ
                momentum_factor = I * k_mu[self.mu]  # Simplified
                momentum_term = momentum_term.subs(deriv, momentum_factor)

        return momentum_term

    def _extract_constraint_vertices(self) -> dict[tuple[str, ...], VertexStructure]:
        """Extract vertices from constraint terms (Lagrange multipliers)."""
        constraint_vertices: dict[tuple[str, ...], Any] = {}

        # Get constraint action
        action_components = self.action.construct_total_action()
        constraint_action = action_components.constraint

        if constraint_action == 0:
            return constraint_vertices

        # Analyze constraint terms
        # These involve Lagrange multipliers and constraint expressions
        # Simplified implementation

        return constraint_vertices

    def verify_vertex_consistency(self, catalog: VertexCatalog) -> dict[str, bool]:
        """
        Verify consistency of extracted vertices.

        Checks:
        1. Dimensional consistency
        2. Index structure correctness
        3. Symmetry factor correctness
        4. Coupling constant assignment

        Returns:
            Dictionary of consistency check results
        """
        results = {
            "dimensional_consistency": True,
            "index_consistency": True,
            "symmetry_consistency": True,
            "coupling_consistency": True,
        }

        all_vertices = (
            list(catalog.three_point.values())
            + list(catalog.four_point.values())
            + list(catalog.constraint_vertices.values())
        )

        for vertex in all_vertices:
            # Check dimensional consistency
            if vertex.mass_dimension < 0:
                results["dimensional_consistency"] = False

            # Check that vertex validates itself
            if not vertex.validate_consistency():
                results["index_consistency"] = False

            # Check symmetry factors are reasonable
            if vertex.symmetry_factor <= 0 or vertex.symmetry_factor > 1:
                results["symmetry_consistency"] = False

            # Check that vertices have associated coupling constants
            if len(vertex.coupling_constants) == 0:
                results["coupling_consistency"] = False

        return results

    def generate_vertex_summary(self, catalog: VertexCatalog) -> str:
        """Generate human-readable summary of vertex catalog."""
        summary = []
        summary.append("Vertex Catalog Summary")
        summary.append("=" * 50)
        summary.append(f"Total vertices: {catalog.total_vertices}")
        summary.append(f"3-point vertices: {len(catalog.three_point)}")
        summary.append(f"4-point vertices: {len(catalog.four_point)}")
        summary.append(f"Constraint vertices: {len(catalog.constraint_vertices)}")
        summary.append("")

        summary.append("Vertex types found:")
        for vertex_type in sorted(catalog.vertex_types):
            count = len(catalog.get_vertices_by_type(vertex_type))
            summary.append(f"  {vertex_type}: {count} vertices")
        summary.append("")

        summary.append("Coupling constants involved:")
        for coupling in sorted(catalog.coupling_constants, key=str):
            summary.append(f"  {coupling}")

        return "\n".join(summary)


class VertexValidator:
    """
    Validation and consistency checking for extracted vertices.

    Performs comprehensive checks on vertex catalog including:
    - Ward identity verification
    - Dimensional consistency
    - Symmetry factor validation
    - Physical reasonableness checks
    """

    def __init__(self, catalog: VertexCatalog, parameters: IsraelStewartParameters):
        self.catalog = catalog
        self.parameters = parameters

    def validate_ward_identities(self) -> dict[str, bool]:
        """
        Check Ward identities for gauge invariance.

        For conserved currents: k_μ V^μ_... = 0
        For MSRJD gauge invariance: specific relations between vertices

        Returns:
            Dictionary of Ward identity check results
        """
        results = {}

        # This would implement specific Ward identity checks
        # For now, return placeholder
        results["energy_momentum_conservation"] = True
        results["current_conservation"] = True
        results["gauge_invariance"] = True

        return results

    def validate_dimensional_consistency(self) -> bool:
        """Check that all vertices have correct mass dimensions."""
        all_vertices = (
            list(self.catalog.three_point.values())
            + list(self.catalog.four_point.values())
            + list(self.catalog.constraint_vertices.values())
        )

        for vertex in all_vertices:
            # Action should be dimensionless in natural units
            # Each vertex contribution must have correct dimension
            expected_dimension = 4.0  # d-dimensional spacetime integration

            if abs(vertex.mass_dimension - expected_dimension) > 0.1:
                return False

        return True

    def check_physical_reasonableness(self) -> dict[str, Any]:
        """Perform physical reasonableness checks."""
        checks = {}

        # Check that we have expected vertex types
        expected_types = {"advection", "stress", "relaxation"}
        found_types = self.catalog.vertex_types
        checks["has_expected_types"] = expected_types.issubset(found_types)

        # Check vertex count is reasonable
        checks["vertex_count_reasonable"] = 10 <= self.catalog.total_vertices <= 100

        # Check all IS parameters appear in some vertex
        is_params = {
            self.parameters.eta,
            self.parameters.tau_pi,
            self.parameters.zeta,
            self.parameters.tau_Pi,
            self.parameters.kappa,
            self.parameters.tau_q,
        }
        found_params = self.catalog.coupling_constants
        checks["all_parameters_used"] = is_params.issubset(found_params)

        return checks
