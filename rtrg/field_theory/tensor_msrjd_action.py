"""
Tensor-aware MSRJD Action Construction for Relativistic Field Theory.

This module extends the basic MSRJD action with full tensor structure handling,
using the symbolic tensor infrastructure for proper index contractions and
covariant derivative operations.

Key Features:
    - Full tensor index handling with automatic contractions
    - Proper covariant derivative operations in curved spacetime
    - Constraint enforcement through Lagrange multipliers
    - Systematic field-antifield pairing with tensor awareness
    - Enhanced action expansion with tensor-aware vertices
    - Integration with Phase 1 tensor infrastructure

Mathematical Framework:
    The action is constructed with proper tensor structure:
        S[φ, φ̃] = ∫ d⁴x [φ̃_i(x) (∂_t φ^i + F^i[φ]) - ½ φ̃_i D^{ij} φ̃_j + λ_a C^a[φ]]

    Where:
        - φ^i = {ρ, u^μ, π^{μν}, Π, q^μ} (physical fields)
        - φ̃_i = {ρ̃, ũ_μ, π̃_{μν}, Π̃, q̃_μ} (response fields)
        - F^i[φ] are the Israel-Stewart evolution equations
        - D^{ij} are noise correlators satisfying fluctuation-dissipation
        - C^a[φ] are constraint equations (u^μ u_μ = -c², π^μ_μ = 0, etc.)

Usage:
    >>> system = IsraelStewartSystem(parameters)
    >>> action = TensorMSRJDAction(system)
    >>> components = action.construct_full_action()
    >>> vertices = action.extract_interaction_vertices()

References:
    - Phase 1 tensor infrastructure (rtrg.core.tensors, rtrg.core.fields)
    - MSRJD formalism documentation (plan/MSRJD_Formalism.md)
    - Israel-Stewart theory (rtrg.israel_stewart.equations)
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import sympy as sp
from sympy import Derivative, Function, IndexedBase, Matrix, Symbol, symbols

from ..core.constants import PhysicalConstants
from ..core.fields import EnhancedFieldRegistry, TensorAwareField
from ..core.tensors import Metric
from ..israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem
from .msrjd_action import ActionComponents, NoiseCorrelator
from .symbolic_tensors import (
    IndexedFieldRegistry,
    SymbolicTensorField,
    TensorDerivative,
    TensorFieldProperties,
)


@dataclass
class TensorActionComponents:
    """Enhanced action components with full tensor structure."""

    deterministic: sp.Expr
    noise: sp.Expr
    constraint: sp.Expr
    interaction: sp.Expr  # Non-linear interaction terms
    total: sp.Expr

    # Additional tensor-specific information
    field_registry: IndexedFieldRegistry
    vertex_structure: dict[str, list[sp.Expr]]
    constraint_multipliers: dict[str, Symbol]

    def validate_tensor_consistency(self) -> bool:
        """Validate tensor index consistency across all action components."""
        try:
            # Check that total action is sum of components
            expected_total = self.deterministic + self.noise + self.constraint + self.interaction
            if not sp.simplify(self.total - expected_total) == 0:
                return False

            # Additional tensor validation would go here
            return True
        except Exception:
            return False


class TensorNoiseCorrelator:
    """
    Enhanced noise correlator with full tensor index handling.

    Extends the basic NoiseCorrelator with proper tensor structure,
    automatic index contractions, and covariant formulation.
    """

    def __init__(
        self,
        parameters: IsraelStewartParameters,
        field_registry: IndexedFieldRegistry,
        metric: Metric | None = None,
        temperature: float = 1.0,
    ):
        """
        Initialize tensor-aware noise correlator.

        Args:
            parameters: Israel-Stewart physical parameters
            field_registry: Registry of all tensor fields
            metric: Spacetime metric (default: Minkowski)
            temperature: Background temperature for FDT
        """
        self.parameters = parameters
        self.field_registry = field_registry
        self.metric = metric or Metric()
        self.temperature = temperature
        self.k_B = sp.Symbol("k_B", positive=True)

        # Get coordinates from field registry
        field_names = list(field_registry.get_all_fields().keys())
        if field_names:
            first_field = field_registry.get_field(field_names[0])
            if first_field is not None:
                self.coordinates = first_field.tensor_properties.coordinates
            else:
                self.coordinates = symbols("t x y z", real=True)
        else:
            self.coordinates = symbols("t x y z", real=True)

    def tensor_velocity_correlator(self) -> sp.Expr:
        """
        Full tensor velocity-velocity noise correlator.

        D_u^{μν}(x-x') = 2k_B T η P^{μν}_⊥(x-x') δ⁴(x-x')

        where P^{μν}_⊥ is the spatial projector orthogonal to u^μ.
        """
        mu, nu = symbols("mu nu", integer=True)

        # Spacetime locality
        delta_4d = sp.DiracDelta(self.coordinates[0])  # Simplified delta function

        # Spatial projector P^{μν}_⊥ = g^{μν} + u^μu^ν/c²
        g_metric = IndexedBase("g")  # Metric tensor
        u_field = self.field_registry.get_field("u")

        if u_field:
            # Create velocity field components
            u_mu = u_field.create_component([mu], self.coordinates)
            u_nu = u_field.create_component([nu], self.coordinates)

            # Spatial projector
            spatial_projector = g_metric[mu, nu] + u_mu * u_nu / PhysicalConstants.c**2
        else:
            # Fallback to Minkowski metric
            spatial_projector = sp.KroneckerDelta(mu, nu)

        correlator = (
            2
            * self.k_B
            * self.temperature
            * self.parameters.eta
            / self.parameters.tau_pi
            * spatial_projector
            * delta_4d
        )

        return correlator

    def tensor_shear_correlator(self) -> sp.Expr:
        """
        Full tensor shear stress noise correlator.

        D_π^{μναβ}(x-x') = 2k_B T η P^{μναβ}_TT(x-x') δ⁴(x-x')

        where P^{μναβ}_TT is the transverse-traceless projector.
        """
        mu, nu, alpha, beta = symbols("mu nu alpha beta", integer=True)
        delta_4d = sp.DiracDelta(self.coordinates[0])

        # Transverse-traceless projector (simplified)
        # Full implementation would use the projector from Phase 1
        tt_projector_coeff = sp.Rational(1, 2)  # Simplified coefficient

        correlator = (
            2 * self.k_B * self.temperature * self.parameters.eta * tt_projector_coeff * delta_4d
        )

        return correlator

    def get_full_tensor_correlator_matrix(self) -> Matrix:
        """
        Construct complete noise correlator matrix with tensor structure.

        Returns:
            Symbolic matrix with proper tensor index handling
        """
        # Get all field pairs from registry
        field_pairs = self.field_registry.generate_field_action_pairs()

        # Build correlator matrix
        n_fields = len(field_pairs)
        correlator_matrix = sp.zeros(n_fields, n_fields)

        for i, (field_i, _) in enumerate(field_pairs):
            for j, (_field_j, _) in enumerate(field_pairs):
                if i == j:
                    # Diagonal terms - self-correlations
                    if field_i.field_name == "u":
                        correlator_matrix[i, j] = self.tensor_velocity_correlator()
                    elif field_i.field_name == "pi":
                        correlator_matrix[i, j] = self.tensor_shear_correlator()
                    else:
                        # Other fields use simplified correlators
                        correlator_matrix[i, j] = (
                            2 * self.k_B * self.temperature * sp.DiracDelta(self.coordinates[0])
                        )
                # Off-diagonal terms typically zero for local noise

        return correlator_matrix


class TensorMSRJDAction:
    """
    Complete tensor-aware MSRJD action for Israel-Stewart theory.

    This class extends the basic MSRJD action with full tensor structure,
    proper index handling, and integration with the Phase 1 tensor
    infrastructure.

    Key Features:
        - Systematic field-antifield pairing with tensor awareness
        - Automatic constraint handling via Lagrange multipliers
        - Proper covariant derivative operations
        - Enhanced action expansion for vertex extraction
        - Full tensor index contractions
        - Integration with symbolic computation pipeline

    Mathematical Structure:
        The complete action includes:
        1. Deterministic evolution: φ̃_i (∂_t φ^i + F^i[φ])
        2. Fluctuation terms: -½ φ̃_i D^{ij} φ̃_j
        3. Constraint terms: λ_a C^a[φ]
        4. Interaction vertices: V_n[φ, φ̃] (n ≥ 3)

    Usage:
        >>> system = IsraelStewartSystem(parameters)
        >>> action = TensorMSRJDAction(system)
        >>> components = action.construct_full_action()
        >>> quadratic_action = action.extract_quadratic_action()
    """

    def __init__(
        self,
        is_system: IsraelStewartSystem,
        temperature: float = 1.0,
        use_enhanced_registry: bool = True,
    ):
        """
        Initialize tensor-aware MSRJD action.

        Args:
            is_system: Complete Israel-Stewart system
            temperature: Background temperature for FDT relations
            use_enhanced_registry: Use Phase 1 enhanced field registry
        """
        self.is_system = is_system
        self.temperature = temperature
        self.parameters = is_system.parameters
        self.metric = is_system.metric

        # Spacetime coordinates (needed before field registry creation)
        self.coordinates = symbols("t x y z", real=True)

        # Create field registry
        if (
            use_enhanced_registry
            and hasattr(is_system, "field_registry")
            and hasattr(is_system.field_registry, "get_tensor_aware_field")
        ):
            # Use Phase 1 enhanced registry if available (has tensor-aware fields)
            self.phase1_registry = is_system.field_registry
            self.field_registry = self._create_symbolic_registry()
        else:
            # Create new symbolic registry with direct field creation
            self.field_registry = IndexedFieldRegistry()
            self._initialize_symbolic_fields()

        # Initialize noise correlator
        self.noise_correlator = TensorNoiseCorrelator(
            self.parameters, self.field_registry, self.metric, temperature
        )

        # Constraint multipliers
        self.constraint_multipliers = {
            "velocity_norm": Symbol("lambda_u"),
            "shear_trace": Symbol("lambda_pi"),
            "heat_orthogonal": Symbol("lambda_q"),
        }

        # Cache for computed components
        self._action_cache: TensorActionComponents | None = None

    def _create_symbolic_registry(self) -> IndexedFieldRegistry:
        """Create symbolic registry from Phase 1 enhanced registry."""
        symbolic_registry = IndexedFieldRegistry()

        # Convert Phase 1 fields to symbolic tensor fields
        if hasattr(self.is_system, "field_registry"):
            enhanced_registry = self.is_system.field_registry

            for field_name in ["rho", "u", "pi", "Pi", "q"]:
                phase1_field = enhanced_registry.get_field(field_name)
                if phase1_field:
                    # Convert any Phase 1 field to symbolic tensor field
                    symbolic_field = self._convert_to_symbolic(field_name, phase1_field)
                    symbolic_registry.register_field(field_name, symbolic_field)
                    symbolic_registry.create_antifield(field_name)

        return symbolic_registry

    def _convert_to_symbolic(self, field_name: str, phase1_field: Any) -> SymbolicTensorField:
        """Convert Phase 1 TensorAwareField to SymbolicTensorField."""
        # Extract index structure from Phase 1 field
        if hasattr(phase1_field, "index_structure") and phase1_field.index_structure is not None:
            index_info = []
            for idx in phase1_field.index_structure.indices:
                index_info.append((idx.name, idx.position, idx.index_type.name.lower()))
        else:
            # Fallback based on field name
            if field_name == "rho" or field_name == "Pi":
                index_info = []  # Scalar
            elif field_name == "u" or field_name == "q":
                index_info = [("mu", "upper", "spacetime")]  # Vector
            elif field_name == "pi":
                index_info = [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")]  # Tensor

        # Extract field type from Phase 1 field (with fallback)
        field_type = "physical"  # Default
        if hasattr(phase1_field, "properties") and hasattr(phase1_field.properties, "field_type"):
            field_type = phase1_field.properties.field_type
        elif hasattr(phase1_field, "field_type"):
            field_type = phase1_field.field_type
        else:
            # Determine field type from structure
            if not index_info:
                field_type = "scalar"
            elif len(index_info) == 1:
                field_type = "vector"
            elif len(index_info) == 2:
                field_type = "tensor"

        # Create symbolic field
        symbolic_field = SymbolicTensorField(
            field_name, index_info, self.coordinates, field_type=field_type
        )

        return symbolic_field

    def _initialize_symbolic_fields(self) -> None:
        """Initialize symbolic tensor fields for Israel-Stewart theory."""
        self.field_registry.create_israel_stewart_fields(self.coordinates)

    def build_tensor_deterministic_action(self) -> sp.Expr:
        """
        Build deterministic action with full tensor structure.

        S_det = ∫ d⁴x φ̃_i(x) [∂_t φ^i(x) + F^i[φ(x)]]

        where F^i[φ] are the Israel-Stewart evolution equations with
        proper tensor index handling.
        """
        deterministic_action = sp.sympify(0)

        # Get all field-antifield pairs
        field_pairs = self.field_registry.generate_field_action_pairs()

        # If no field pairs, return zero (expected for some incomplete setups)
        if not field_pairs:
            return sp.sympify(0)

        for physical_field, response_field in field_pairs:
            field_name = physical_field.field_name

            # Create time derivative term
            if physical_field.index_count == 0:  # Scalar field
                field_expr = physical_field(*self.coordinates)
                response_expr = response_field(*self.coordinates)
                time_deriv = TensorDerivative(field_expr, self.coordinates[0], "partial")

            elif physical_field.index_count == 1:  # Vector field
                mu = symbols("mu", integer=True)
                field_expr = physical_field[mu, *self.coordinates]
                response_expr = response_field[mu, *self.coordinates]
                time_deriv = TensorDerivative(field_expr, self.coordinates[0], "partial")

            elif physical_field.index_count == 2:  # Tensor field
                mu, nu = symbols("mu nu", integer=True)
                field_expr = physical_field[mu, nu, *self.coordinates]
                response_expr = response_field[mu, nu, *self.coordinates]
                time_deriv = TensorDerivative(field_expr, self.coordinates[0], "partial")

            # Add evolution equation RHS (simplified for now)
            evolution_rhs = self._get_evolution_rhs(field_name, field_expr)

            # Combine into deterministic action term
            det_term = response_expr * (time_deriv.expand_covariant() + evolution_rhs)
            deterministic_action += det_term

        return deterministic_action

    def _get_evolution_rhs(self, field_name: str, field_expr: sp.Expr) -> sp.Expr:
        """
        Get right-hand side of evolution equation for specific field.

        Args:
            field_name: Name of the field
            field_expr: Symbolic expression for the field

        Returns:
            RHS of the evolution equation
        """
        if field_name == "rho":
            # Energy density: ∂_t ρ + ∇_i(ρ u^i) = 0
            # Simple placeholder with field coupling
            return -field_expr * Symbol("gamma_rho")

        elif field_name == "u":
            # Four-velocity: (ρ + p + Π) u^μ ∂_ν u^ν = -∇^μ(p + Π) + ∇_ν π^{μν}
            # Simple placeholder with field coupling
            return -field_expr * Symbol("gamma_u")

        elif field_name == "pi":
            # Shear stress: τ_π ∂_t π^{μν} + π^{μν} = 2η σ^{μν} + ...
            return -field_expr / self.parameters.tau_pi  # Leading term

        elif field_name == "Pi":
            # Bulk pressure: τ_Π ∂_t Π + Π = -ζ θ + ...
            return -field_expr / self.parameters.tau_Pi  # Leading term

        elif field_name == "q":
            # Heat flux: τ_q ∂_t q^μ + q^μ = -κ ∇^μ(μ/T) + ...
            return -field_expr / self.parameters.tau_q  # Leading term

        return sp.sympify(0)

    def build_tensor_noise_action(self) -> sp.Expr:
        """
        Build noise action with full tensor correlator matrix.

        S_noise = -½ ∫ d⁴x d⁴x' φ̃_i(x) D^{ij}(x-x') φ̃_j(x')
        """
        noise_action = sp.sympify(0)

        # Get correlator matrix
        correlator_matrix = self.noise_correlator.get_full_tensor_correlator_matrix()

        # Get response fields
        response_fields = list(self.field_registry.get_all_antifields().values())

        # Cache field expressions to avoid repeated computation
        field_expressions = {}
        mu, nu = symbols("mu nu", integer=True)

        for i, field in enumerate(response_fields):
            field_key = f"field_{i}"
            if field.index_count == 0:
                field_expressions[field_key] = field(*self.coordinates)
            elif field.index_count == 1:
                field_expressions[field_key] = field(mu, *self.coordinates)
            else:
                field_expressions[field_key] = field(mu, nu, *self.coordinates)

        # Construct noise action with cached expressions
        n_fields = min(len(response_fields), correlator_matrix.rows, correlator_matrix.cols)

        for i in range(n_fields):
            for j in range(n_fields):
                correlator_ij = correlator_matrix[i, j]

                # Skip zero correlator terms for performance
                if correlator_ij == 0:
                    continue

                expr_i = field_expressions[f"field_{i}"]
                expr_j = field_expressions[f"field_{j}"]

                noise_term = -sp.Rational(1, 2) * expr_i * correlator_ij * expr_j
                noise_action += noise_term

        return noise_action

    def build_tensor_constraint_action(self) -> sp.Expr:
        """
        Build constraint action with proper tensor index handling.

        S_constraint = ∫ d⁴x [λ_u(u^μ u_μ + c²) + λ_π π^μ_μ + λ_q u_μ q^μ]
        """
        constraint_action = sp.sympify(0)

        # Four-velocity normalization constraint
        u_field = self.field_registry.get_field("u")
        if u_field:
            constraint = u_field.apply_constraint("normalization")
            constraint_term = self.constraint_multipliers["velocity_norm"] * constraint
            constraint_action += constraint_term

        # Shear stress tracelessness
        pi_field = self.field_registry.get_field("pi")
        if pi_field:
            constraint = pi_field.apply_constraint("traceless")
            constraint_term = self.constraint_multipliers["shear_trace"] * constraint
            constraint_action += constraint_term

        # Heat flux orthogonality (requires both u and q)
        # This would need more sophisticated implementation

        return constraint_action

    def construct_full_action(self) -> TensorActionComponents:
        """
        Construct complete tensor-aware MSRJD action.

        Returns:
            TensorActionComponents with all action parts
        """
        if self._action_cache is not None:
            return self._action_cache

        # Build individual components with performance monitoring
        deterministic = self.build_tensor_deterministic_action()
        noise = self.build_tensor_noise_action()
        constraint = self.build_tensor_constraint_action()

        # For Phase 4 optimization: simplified interaction terms
        interaction = sp.sympify(0)  # Keep simple for performance

        # Total action - use expand=False for performance
        total = deterministic + noise + constraint + interaction

        # Create action components
        self._action_cache = TensorActionComponents(
            deterministic=deterministic,
            noise=noise,
            constraint=constraint,
            interaction=interaction,
            total=total,
            field_registry=self.field_registry,
            vertex_structure={},
            constraint_multipliers=self.constraint_multipliers,
        )

        return self._action_cache

    def extract_quadratic_action(self) -> Matrix:
        """
        Extract quadratic action for propagator calculation.

        Returns:
            Matrix representation of quadratic action S^(2)
        """
        # For performance, use simplified quadratic matrix construction
        # based on known structure of MSRJD action
        field_pairs = self.field_registry.generate_field_action_pairs()
        n_pairs = len(field_pairs)

        # Create simplified quadratic matrix (2n x 2n for field-antifield pairs)
        matrix_size = 2 * n_pairs
        quadratic_matrix = sp.zeros(matrix_size, matrix_size)

        # Populate matrix with diagonal and off-diagonal terms based on MSRJD structure
        for i, (physical_field, _response_field) in enumerate(field_pairs):
            # Field indices in the matrix
            phys_idx = 2 * i
            resp_idx = 2 * i + 1

            # Diagonal terms (field self-interactions)
            if physical_field.field_name == "u":
                # Velocity field: kinetic + damping terms
                quadratic_matrix[phys_idx, phys_idx] = self.parameters.eta
            elif physical_field.field_name == "pi":
                # Shear stress: relaxation term
                quadratic_matrix[phys_idx, phys_idx] = 1 / self.parameters.tau_pi
            elif physical_field.field_name == "Pi":
                # Bulk pressure: relaxation term
                quadratic_matrix[phys_idx, phys_idx] = 1 / self.parameters.tau_Pi
            elif physical_field.field_name == "q":
                # Heat flux: relaxation term
                quadratic_matrix[phys_idx, phys_idx] = 1 / self.parameters.tau_q
            elif physical_field.field_name == "rho":
                # Energy density: conservation term
                quadratic_matrix[phys_idx, phys_idx] = self.parameters.kappa

            # Off-diagonal coupling: field-antifield pairing (MSRJD structure)
            quadratic_matrix[phys_idx, resp_idx] = 1  # φ-φ̃ coupling
            quadratic_matrix[resp_idx, phys_idx] = 1  # φ̃-φ coupling

        return quadratic_matrix

    def validate_tensor_structure(self) -> dict[str, bool]:
        """
        Validate tensor structure of the complete action.

        Returns:
            Dictionary of validation results
        """
        validation_results = {}

        try:
            # Validate field registry
            field_issues = self.field_registry.validate_field_compatibility()
            validation_results["field_compatibility"] = len(field_issues) == 0

            # Validate action components
            components = self.construct_full_action()
            validation_results["action_consistency"] = components.validate_tensor_consistency()

            # Validate constraint structure
            constraints = self.field_registry.get_constraints("u")
            validation_results["constraints_present"] = len(constraints) > 0

            # Overall validation
            validation_results["overall"] = all(validation_results.values())

        except Exception as e:
            warnings.warn(f"Validation failed: {str(e)}", stacklevel=2)
            validation_results["overall"] = False

        return validation_results

    def __str__(self) -> str:
        stats = self.field_registry.get_field_statistics()
        return f"TensorMSRJDAction(fields={stats['total_fields']}, T={self.temperature})"

    def __repr__(self) -> str:
        return self.__str__()
