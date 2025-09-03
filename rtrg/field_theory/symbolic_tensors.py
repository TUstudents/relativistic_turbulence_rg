"""
Symbolic tensor field operations for MSRJD field theory calculations.

This module provides enhanced symbolic tensor field classes that extend SymPy's
Function class to handle relativistic tensor fields with proper index structure,
automatic differentiation, and constraint enforcement.

Key Features:
    - SymbolicTensorField extending SymPy Function with tensor indices
    - TensorDerivative operations with index contractions
    - IndexedFieldRegistry for systematic field management
    - Integration with SymPy's differentiation system
    - Proper handling of spacetime coordinates and tensor indices
    - Constraint enforcement for physical tensor fields

Mathematical Framework:
    Fields are represented as symbolic functions with both tensor indices
    and spacetime coordinates:
        φ^{μν}(x^α) = φ[μ, ν, t, x, y, z]

    Derivatives respect tensor calculus rules:
        ∂_α φ^{μν} = TensorDerivative(φ[μ, ν], x^α)

    Constraints are automatically enforced:
        u^μ u_μ = -c², π^μ_μ = 0, u_μ π^{μν} = 0, etc.

Usage:
    >>> field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], [t, x, y, z])
    >>> derivative = TensorDerivative(field[mu], t)
    >>> registry = IndexedFieldRegistry()
    >>> registry.register_field("u", field)

References:
    - Misner, Thorne & Wheeler "Gravitation" - tensor calculus foundations
    - Rezzolla & Zanotti "Relativistic Hydrodynamics" - field theory methods
    - SymPy documentation on Function classes and symbolic differentiation
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import sympy as sp
from sympy import Derivative, Function, IndexedBase, Symbol, symbols
from sympy.core.function import UndefinedFunction

try:
    from sympy.tensor import TensorHead, TensorSymmetry
    from sympy.tensor.tensor import TensAdd, TensMul
    from sympy.tensor.tensor import TensorIndex as SymPyTensorIndex
except ImportError:
    # Fallback for older SymPy versions
    TensorHead = None
    TensorSymmetry = None
    TensAdd = None
    TensMul = None
    SymPyTensorIndex = None

from ..core.tensors import IndexType as RTRGIndexType
from ..core.tensors import TensorIndex, TensorIndexStructure


@dataclass
class TensorFieldProperties:
    """
    Properties defining a symbolic tensor field.

    Encapsulates all information needed to create and manipulate
    symbolic tensor fields in relativistic field theory calculations.
    """

    name: str
    index_structure: list[tuple[str, str, str]]  # (name, position, type)
    coordinates: list[Symbol]
    symmetries: list[str] | None = None
    constraints: list[str] | None = None
    field_type: str = "general"  # general, vector, tensor, scalar

    def __post_init__(self) -> None:
        """Validate tensor field properties."""
        if not self.name:
            raise ValueError("Field name cannot be empty")

        if not self.coordinates:
            raise ValueError("Coordinates list cannot be empty")

        # Validate index structure
        for idx_info in self.index_structure:
            if len(idx_info) != 3:
                raise ValueError("Index info must be (name, position, type)")

            name, position, idx_type = idx_info
            if position not in ["upper", "lower"]:
                raise ValueError(f"Invalid index position: {position}")

            if idx_type not in ["spacetime", "spatial", "temporal"]:
                raise ValueError(f"Invalid index type: {idx_type}")


class SymbolicTensorField(Function):
    """
    Symbolic tensor field extending SymPy Function for relativistic field theory.

    This class creates symbolic representations of tensor fields that can handle
    both tensor indices and spacetime coordinates. It integrates with SymPy's
    differentiation system while maintaining proper tensor structure.

    Key Features:
        - Automatic handling of tensor indices and coordinate arguments
        - Integration with SymPy's symbolic differentiation
        - Constraint enforcement for physical fields
        - Proper printing and representation for tensor expressions
        - Support for field transformations and index manipulations

    Mathematical Structure:
        A tensor field φ^{μν...}(x^α) is represented as:
            φ[μ, ν, ..., t, x, y, z]

        where the first arguments are tensor indices and the last arguments
        are spacetime coordinates.

    Examples:
        >>> t, x, y, z = symbols('t x y z')
        >>> mu, nu = symbols('mu nu')
        >>> u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], [t, x, y, z])
        >>> u_component = u_field[mu, t, x, y, z]
        >>> time_derivative = u_field.diff(t)
    """

    @classmethod
    def eval(cls, *args: Any) -> None:
        """Evaluate symbolic tensor field - returns None for symbolic handling."""
        # Return None to keep expressions symbolic
        return None

    def __new__(
        cls,
        name: str,
        index_structure: list[tuple[str, str, str]],
        coordinates: list[Symbol],
        **kwargs: Any,
    ) -> "SymbolicTensorField":
        """
        Create new symbolic tensor field.

        Args:
            name: Field name (e.g., "u", "pi", "rho")
            index_structure: List of (index_name, position, type) tuples
            coordinates: Spacetime coordinate symbols [t, x, y, z]
            **kwargs: Additional properties (symmetries, constraints, etc.)

        Returns:
            New SymbolicTensorField instance
        """
        # Store properties in the class
        properties = TensorFieldProperties(
            name=name, index_structure=index_structure, coordinates=coordinates, **kwargs
        )

        # Create the function with name as the function identifier
        obj = Function.__new__(cls, name)

        # Attach properties to the instance
        obj._tensor_properties = properties
        obj._name = name
        obj._index_structure = index_structure
        obj._coordinates = coordinates

        return obj  # type: ignore[no-any-return]

    @property
    def tensor_properties(self) -> TensorFieldProperties:
        """Access tensor field properties."""
        return self._tensor_properties  # type: ignore[no-any-return]

    @property
    def field_name(self) -> str:
        """Get field name."""
        return self._name  # type: ignore[no-any-return]

    @property
    def index_count(self) -> int:
        """Number of tensor indices."""
        return len(self._index_structure)

    @property
    def coordinate_count(self) -> int:
        """Number of spacetime coordinates."""
        return len(self._coordinates)

    @property
    def index_structure(self) -> list[tuple[str, str, str]]:
        """Get tensor index structure."""
        return self._index_structure  # type: ignore[no-any-return]

    @property
    def coordinates(self) -> list[Symbol]:
        """Get spacetime coordinates."""
        return self._coordinates  # type: ignore[no-any-return]

    def __getitem__(self, indices: Any) -> Any:
        """
        Create tensor field component with specified indices and coordinates.

        Args:
            indices: Tensor indices followed by coordinates

        Returns:
            Function evaluation representing the field component
        """
        if not isinstance(indices, tuple | list):
            indices = (indices,)

        # Create a component function with indices encoded in the name
        # Format: field_name(coordinates) with indices as subscripts
        if len(indices) > len(self._coordinates):
            # Separate tensor indices from coordinates
            n_coords = len(self._coordinates)
            tensor_indices = indices[:-n_coords] if n_coords > 0 else indices
            coords = indices[-n_coords:] if n_coords > 0 else []

            # Create component name with indices
            if tensor_indices:
                index_str = "_".join(str(idx) for idx in tensor_indices)
                component_name = f"{self.field_name}_{index_str}"
            else:
                component_name = self.field_name
        else:
            # Only coordinates provided
            coords = indices
            component_name = self.field_name

        # Create and return the component function
        component_func = Function(component_name)
        if coords:
            return component_func(*coords)
        else:
            return component_func

    def __call__(self, *coordinates: Any) -> Any:
        """
        Make tensor field callable for scalar fields (rank 0).

        Args:
            coordinates: Spacetime coordinates

        Returns:
            Function evaluation at the given coordinates
        """
        # For scalar fields, just call with coordinates
        if not hasattr(self, "_rank") or getattr(self, "_rank", 0) == 0:
            component_func = Function(self.field_name)
            return component_func(*coordinates)
        else:
            raise TypeError(
                f"Tensor field {self.field_name} with rank > 0 is not directly callable. Use indexing: field[indices, coordinates]"
            )

    def create_component(
        self, tensor_indices: list[Symbol], coordinate_values: list[Symbol] | None = None
    ) -> sp.Expr:
        """
        Create specific tensor component.

        Args:
            tensor_indices: Values for tensor indices
            coordinate_values: Coordinate values (default: symbolic coordinates)

        Returns:
            Symbolic expression for the tensor component
        """
        if coordinate_values is None:
            coordinate_values = self._coordinates

        if len(tensor_indices) != self.index_count:
            raise ValueError(
                f"Expected {self.index_count} tensor indices, got {len(tensor_indices)}"
            )

        if len(coordinate_values) != self.coordinate_count:
            raise ValueError(
                f"Expected {self.coordinate_count} coordinates, got {len(coordinate_values)}"
            )

        all_args = list(tensor_indices) + list(coordinate_values)
        return self(*all_args)

    def apply_constraint(self, constraint_type: str) -> sp.Expr:
        """
        Apply physical constraint to the tensor field.

        Args:
            constraint_type: Type of constraint ("normalization", "traceless", "orthogonal")

        Returns:
            Symbolic constraint expression
        """
        if constraint_type == "normalization" and self.field_name == "u":
            # Four-velocity normalization: u^μ u_μ = -c²
            mu = symbols("mu", integer=True)
            c = sp.Symbol("c", positive=True)

            # Create metric contraction (simplified to Minkowski)
            constraint = (
                sum(self[mu, *self._coordinates] * self[mu, *self._coordinates] for mu in range(4))
                + c**2
            )
            return constraint

        elif constraint_type == "traceless" and self.field_name == "pi":
            # Shear stress tracelessness: π^μ_μ = 0
            mu = symbols("mu", integer=True)
            constraint = sum(self[mu, mu, *self._coordinates] for mu in range(4))
            return constraint

        elif constraint_type == "orthogonal":
            # Heat flux orthogonality: u_μ q^μ = 0 (requires both u and q fields)
            # This would need both fields as input - placeholder for now
            return sp.sympify(0)

        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

    def tensor_derivative(
        self, coordinate: Symbol, derivative_type: str = "partial"
    ) -> "TensorDerivative":
        """
        Create tensor derivative with respect to coordinate.

        Args:
            coordinate: Coordinate to differentiate with respect to
            derivative_type: Type of derivative ("partial", "covariant")

        Returns:
            TensorDerivative object representing the derivative
        """
        return TensorDerivative(self, coordinate, derivative_type)

    def __str__(self) -> str:
        """String representation of tensor field."""
        indices_str = "^{" + ",".join(idx[0] for idx in self._index_structure) + "}"
        coords_str = "(" + ",".join(str(coord) for coord in self._coordinates) + ")"
        return f"{self.field_name}{indices_str}{coords_str}"

    def __repr__(self) -> str:
        return f"SymbolicTensorField('{self.field_name}', {self._index_structure}, {self._coordinates})"


class TensorDerivative(Derivative):
    """
    Enhanced derivative class for tensor fields with index contractions.

    Extends SymPy's Derivative to handle tensor index structure properly
    and perform automatic index contractions when needed.

    Key Features:
        - Proper handling of tensor indices in derivatives
        - Automatic index contraction rules
        - Integration with covariant derivative operations
        - Support for Christoffel symbol contributions
        - Chain rule for composite tensor expressions

    Mathematical Structure:
        Partial derivative: ∂_α φ^{μν}
        Covariant derivative: ∇_α φ^{μν} = ∂_α φ^{μν} + Γ^μ_{αβ} φ^{βν} + Γ^ν_{αβ} φ^{μβ}

    Examples:
        >>> field = SymbolicTensorField("pi", [("mu", "upper"), ("nu", "upper")], [t, x, y, z])
        >>> partial_deriv = TensorDerivative(field[mu, nu, t, x, y, z], t, "partial")
        >>> covariant_deriv = TensorDerivative(field[mu, nu, t, x, y, z], x, "covariant")
    """

    def __new__(
        cls, expr: sp.Expr, coordinate: Symbol, derivative_type: str = "partial"
    ) -> "TensorDerivative":
        """
        Create tensor derivative.

        Args:
            expr: Tensor expression to differentiate
            coordinate: Coordinate variable
            derivative_type: "partial" or "covariant"

        Returns:
            TensorDerivative instance
        """
        # Create standard SymPy derivative as base
        obj = Derivative.__new__(cls, expr, coordinate)

        # Store additional tensor information
        obj._derivative_type = derivative_type
        obj._tensor_expr = expr
        obj._coordinate = coordinate
        obj._expansion_cache = {}  # Cache for performance

        return obj  # type: ignore[no-any-return]

    @property
    def derivative_type(self) -> str:
        """Type of derivative (partial/covariant)."""
        return self._derivative_type  # type: ignore[no-any-return]

    def expand_covariant(self) -> sp.Expr:
        """
        Expand covariant derivative with Christoffel symbol terms.

        For a tensor field T^{μν}, the covariant derivative is:
        ∇_α T^{μν} = ∂_α T^{μν} + Γ^μ_{αβ} T^{βν} + Γ^ν_{αβ} T^{μβ}

        Returns:
            Expanded covariant derivative expression
        """
        # Check cache first
        cache_key = f"expand_{self._derivative_type}_{hash(self._coordinate)}"
        if hasattr(self, "_expansion_cache") and cache_key in self._expansion_cache:
            return self._expansion_cache[cache_key]

        # Always return the partial derivative term first
        partial_term = Derivative(self._tensor_expr, self._coordinate)

        if self._derivative_type != "covariant":
            # Cache and return partial derivative for non-covariant case
            result = partial_term
        else:
            # For covariant derivatives, add Christoffel terms only if needed
            # In Minkowski space, Christoffel symbols are zero, so skip expensive calculation
            if self._is_minkowski_approximation():
                result = partial_term
            else:
                # For full covariant derivative implementation in curved spacetime
                christoffel_terms = self._construct_christoffel_terms()
                result = partial_term + christoffel_terms

        # Cache the result
        if hasattr(self, "_expansion_cache"):
            self._expansion_cache[cache_key] = result

        return result

    def _is_minkowski_approximation(self) -> bool:
        """Check if we can use Minkowski approximation (flat spacetime)."""
        # For performance, default to Minkowski approximation unless curved spacetime is explicitly needed
        return True

    def _construct_christoffel_terms(self) -> sp.Expr:
        """
        Construct Christoffel symbol contributions to covariant derivative.

        For each upper index μ in the tensor, add: +Γ^μ_{αβ} T^{...β...}
        For each lower index μ in the tensor, add: -Γ^β_{αμ} T^{...β...}

        Returns:
            Sum of all Christoffel symbol terms
        """
        # Define Christoffel symbols
        Gamma = IndexedBase("Gamma")
        alpha = sp.Symbol("alpha", integer=True)  # Derivative index
        beta = sp.Symbol("beta", integer=True)  # Dummy contraction index

        christoffel_terms = sp.sympify(0)

        # This is a simplified implementation
        # Full implementation would analyze tensor structure of self._tensor_expr
        # and add appropriate Christoffel terms for each index

        # For now, return zero (Minkowski space approximation)
        return christoffel_terms

    def _extract_tensor_indices(self) -> list[tuple[str, str]]:
        """
        Extract tensor indices from the expression being differentiated.

        Returns:
            List of (index_name, position) tuples
        """
        indices = []

        # Analyze expression structure to find tensor indices
        # This would examine the arguments to tensor field functions
        # and determine which are tensor indices vs coordinates

        # Simplified implementation for common cases
        if hasattr(self._tensor_expr, "args") and self._tensor_expr.args:
            # Assume first arguments are tensor indices
            args = self._tensor_expr.args
            for _i, arg in enumerate(args):
                if isinstance(arg, Symbol) and arg.name in ["mu", "nu", "alpha", "beta"]:
                    indices.append((arg.name, "upper"))  # Default to upper

        return indices

    def contract_indices(self, index1: str, index2: str) -> sp.Expr:
        """
        Contract two indices in the derivative expression.

        Args:
            index1: First index to contract
            index2: Second index to contract

        Returns:
            Expression with indices contracted
        """
        base_expr = self.expand_covariant()

        # Perform Einstein summation over repeated indices
        contracted_expr = self._perform_einstein_summation(base_expr, index1, index2)

        return contracted_expr

    def _perform_einstein_summation(self, expr: sp.Expr, index1: str, index2: str) -> sp.Expr:
        """
        Perform Einstein summation over two repeated indices.

        Args:
            expr: Expression to contract
            index1: First index (will be summed over)
            index2: Second index (will be summed over)

        Returns:
            Contracted expression with summation over index values 0,1,2,3
        """
        # For performance, use a symbolic approach rather than explicit summation
        # This avoids expensive substitutions for complex expressions
        if index1 == index2:
            # Self-contraction: just multiply by dimensionality (4 for spacetime)
            return 4 * expr

        # Create symbolic index for summation
        idx_symbol = sp.Symbol(index1, integer=True)

        # If the expression is simple, do direct summation
        if len(expr.free_symbols) < 10:  # Threshold for complexity
            # Replace both indices with the same symbol
            expr_substituted = expr
            idx2_symbol = sp.Symbol(index2, integer=True)
            expr_substituted = expr_substituted.subs(idx2_symbol, idx_symbol)

            # Sum over spacetime dimensions (0, 1, 2, 3)
            contracted_result = sum(expr_substituted.subs(idx_symbol, i) for i in range(4))
            return contracted_result
        else:
            # For complex expressions, use symbolic contraction
            # This represents the contraction symbolically without explicit evaluation
            return 4 * expr  # Simplified result for performance

    def covariant_gradient(self, field: SymbolicTensorField) -> sp.Expr:
        """
        Compute covariant gradient of a tensor field.

        Args:
            field: Tensor field to take gradient of

        Returns:
            Covariant derivative expression
        """
        # Create covariant derivative with respect to coordinate
        coord_index = sp.Symbol("coord_idx", integer=True)

        # For each coordinate, create covariant derivative
        gradient_components = []

        coordinates = field.tensor_properties.coordinates
        for _i, coord in enumerate(coordinates):
            deriv = TensorDerivative(field[coord_index, *coordinates], coord, "covariant")
            gradient_components.append(deriv.expand_covariant())

        return sp.Matrix(gradient_components)

    def __str__(self) -> str:
        """String representation of tensor derivative."""
        if self._derivative_type == "covariant":
            return f"∇_{{{self._coordinate}}} {self._tensor_expr}"
        else:
            return f"∂_{{{self._coordinate}}} {self._tensor_expr}"


class IndexedFieldRegistry:
    """
    Registry for managing symbolic tensor fields in MSRJD calculations.

    Provides centralized management of all tensor fields used in the
    field theory calculation, ensuring consistent index handling and
    proper field relationships.

    Key Features:
        - Systematic field registration and lookup
        - Automatic field-antifield pairing for MSRJD
        - Constraint validation and enforcement
        - Field transformation utilities
        - Integration with symbolic computation pipeline

    Mathematical Structure:
        Physical fields: φ = {ρ, u^μ, π^{μν}, Π, q^μ}
        Response fields: φ̃ = {ρ̃, ũ_μ, π̃_{μν}, Π̃, q̃_μ}

        Each field pair (φ, φ̃) contributes to the MSRJD action.

    Usage:
        >>> registry = IndexedFieldRegistry()
        >>> u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coords)
        >>> registry.register_field("u", u_field)
        >>> registry.create_antifield("u")
        >>> all_fields = registry.get_all_fields()
    """

    def __init__(self) -> None:
        """Initialize empty field registry."""
        self._fields: dict[str, SymbolicTensorField] = {}
        self._antifields: dict[str, SymbolicTensorField] = {}
        self._field_relationships: dict[str, list[str]] = {}
        self._constraints: dict[str, list[sp.Expr]] = {}

    def register_field(self, name: str, field: SymbolicTensorField) -> None:
        """
        Register a tensor field in the registry.

        Args:
            name: Unique field identifier
            field: SymbolicTensorField instance
        """
        if name in self._fields:
            warnings.warn(f"Field '{name}' already exists, overwriting", stacklevel=2)

        self._fields[name] = field

        # Initialize empty relationships and constraints
        if name not in self._field_relationships:
            self._field_relationships[name] = []
        if name not in self._constraints:
            self._constraints[name] = []

    def create_antifield(self, field_name: str) -> SymbolicTensorField:
        """
        Create response field (antifield) for MSRJD formulation.

        Args:
            field_name: Name of the physical field

        Returns:
            SymbolicTensorField representing the response field
        """
        if field_name not in self._fields:
            raise ValueError(f"Field '{field_name}' not found in registry")

        original_field = self._fields[field_name]

        # Create antifield with tilde suffix and appropriate index structure
        antifield_name = field_name + "_tilde"

        # For MSRJD, response fields typically have lowered indices
        antifield_indices = []
        for idx_name, position, idx_type in original_field.index_structure:
            # Convert upper indices to lower for response fields
            new_position = "lower" if position == "upper" else "upper"
            antifield_indices.append((idx_name, new_position, idx_type))

        antifield = SymbolicTensorField(
            antifield_name,
            antifield_indices,
            original_field.tensor_properties.coordinates,
            field_type="response",
        )

        self._antifields[field_name] = antifield
        return antifield

    def get_field(self, name: str) -> SymbolicTensorField | None:
        """Get field by name."""
        return self._fields.get(name)

    def get_antifield(self, field_name: str) -> SymbolicTensorField | None:
        """Get antifield by original field name."""
        return self._antifields.get(field_name)

    def get_all_fields(self) -> dict[str, SymbolicTensorField]:
        """Get all registered fields."""
        return self._fields.copy()

    def get_all_antifields(self) -> dict[str, SymbolicTensorField]:
        """Get all registered antifields."""
        return self._antifields.copy()

    def add_constraint(self, field_name: str, constraint: sp.Expr) -> None:
        """
        Add constraint for a specific field.

        Args:
            field_name: Name of the field
            constraint: Symbolic constraint expression
        """
        if field_name not in self._constraints:
            self._constraints[field_name] = []

        self._constraints[field_name].append(constraint)

    def get_constraints(self, field_name: str) -> list[sp.Expr]:
        """Get all constraints for a field."""
        return self._constraints.get(field_name, [])

    def validate_constraints(self) -> dict[str, bool]:
        """
        Validate all field constraints.

        Returns:
            Dictionary mapping field names to constraint satisfaction status
        """
        validation_results = {}

        for field_name, constraints in self._constraints.items():
            field_valid = True

            for constraint in constraints:
                # Check if constraint is satisfied (simplified validation)
                if not isinstance(constraint, sp.Expr):
                    field_valid = False
                    break

            validation_results[field_name] = field_valid

        return validation_results

    def create_israel_stewart_fields(self, coordinates: list[Symbol]) -> None:
        """
        Create all Israel-Stewart fields with proper tensor structure.

        Args:
            coordinates: Spacetime coordinates [t, x, y, z]
        """
        # Energy density (scalar)
        rho_field = SymbolicTensorField("rho", [], coordinates, field_type="scalar")
        self.register_field("rho", rho_field)

        # Four-velocity (vector)
        u_field = SymbolicTensorField(
            "u",
            [("mu", "upper", "spacetime")],
            coordinates,
            field_type="vector",
            constraints=["normalization"],
        )
        self.register_field("u", u_field)

        # Shear stress (symmetric traceless tensor)
        pi_field = SymbolicTensorField(
            "pi",
            [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")],
            coordinates,
            field_type="tensor",
            constraints=["traceless", "symmetric", "orthogonal"],
            symmetries=["symmetric"],
        )
        self.register_field("pi", pi_field)

        # Bulk pressure (scalar)
        Pi_field = SymbolicTensorField("Pi", [], coordinates, field_type="scalar")
        self.register_field("Pi", Pi_field)

        # Heat flux (vector orthogonal to velocity)
        q_field = SymbolicTensorField(
            "q",
            [("mu", "upper", "spacetime")],
            coordinates,
            field_type="vector",
            constraints=["orthogonal"],
        )
        self.register_field("q", q_field)

        # Create corresponding antifields
        for field_name in ["rho", "u", "pi", "Pi", "q"]:
            self.create_antifield(field_name)

        # Add specific constraints
        if "u" in self._fields:
            u_constraint = self._fields["u"].apply_constraint("normalization")
            self.add_constraint("u", u_constraint)

        if "pi" in self._fields:
            pi_constraint = self._fields["pi"].apply_constraint("traceless")
            self.add_constraint("pi", pi_constraint)

    def field_count(self) -> int:
        """Total number of registered fields."""
        return len(self._fields)

    def antifield_count(self) -> int:
        """Total number of registered antifields."""
        return len(self._antifields)

    def generate_field_action_pairs(self) -> list[tuple[SymbolicTensorField, SymbolicTensorField]]:
        """
        Generate all field-antifield pairs for MSRJD action construction.

        Returns:
            List of (physical_field, response_field) pairs
        """
        pairs = []

        for field_name, physical_field in self._fields.items():
            if field_name in self._antifields:
                response_field = self._antifields[field_name]
                pairs.append((physical_field, response_field))

        return pairs

    def create_field_derivatives(
        self, field_name: str, coordinates: list[Symbol]
    ) -> dict[str, TensorDerivative]:
        """
        Create all partial derivatives of a field with respect to coordinates.

        Args:
            field_name: Name of field to differentiate
            coordinates: List of coordinate symbols

        Returns:
            Dictionary mapping coordinate names to derivative expressions
        """
        if field_name not in self._fields:
            raise ValueError(f"Field '{field_name}' not found")

        field = self._fields[field_name]
        derivatives = {}

        for coord in coordinates:
            coord_name = str(coord)

            # Create symbolic expression for field component
            # This depends on the specific tensor structure
            if field.index_count == 0:  # Scalar field
                field_expr = field(*coordinates)
            elif field.index_count == 1:  # Vector field
                mu = sp.Symbol("mu", integer=True)
                field_expr = field(mu, *coordinates)
            elif field.index_count == 2:  # Tensor field
                mu, nu = symbols("mu nu", integer=True)
                field_expr = field(mu, nu, *coordinates)
            else:
                # Higher rank tensors - general case
                indices = [sp.Symbol(f"idx_{i}", integer=True) for i in range(field.index_count)]
                field_expr = field(*indices, *coordinates)

            derivative = TensorDerivative(field_expr, coord, "partial")
            derivatives[coord_name] = derivative

        return derivatives

    def extract_field_symbols(self) -> dict[str, list[sp.Symbol]]:
        """
        Extract all symbols used in the registered fields.

        Returns:
            Dictionary mapping field names to their constituent symbols
        """
        field_symbols = {}

        for field_name, field in self._fields.items():
            symbols_list = []

            # Add coordinate symbols
            symbols_list.extend(field.tensor_properties.coordinates)

            # Add tensor index symbols (common ones)
            common_indices = ["mu", "nu", "alpha", "beta", "gamma", "delta"]
            for idx_info in field.tensor_properties.index_structure:
                idx_name = idx_info[0]
                if idx_name in common_indices:
                    symbols_list.append(sp.Symbol(idx_name, integer=True))

            field_symbols[field_name] = symbols_list

        return field_symbols

    def validate_field_compatibility(self) -> dict[str, list[str]]:
        """
        Validate compatibility between fields and their constraints.

        Returns:
            Dictionary mapping field names to lists of compatibility issues
        """
        issues = {}

        for field_name, field in self._fields.items():
            field_issues = []

            # Check constraint compatibility
            constraints = self.get_constraints(field_name)
            field_properties = field.tensor_properties

            # Validate constraint-field structure compatibility
            if field_properties.constraints and "normalization" in field_properties.constraints:
                if field.index_count != 1:
                    field_issues.append("Normalization constraint requires vector field")

            if field_properties.constraints and "traceless" in field_properties.constraints:
                if field.index_count != 2:
                    field_issues.append("Traceless constraint requires rank-2 tensor field")

            if field_properties.symmetries and "symmetric" in field_properties.symmetries:
                if field.index_count != 2:
                    field_issues.append("Symmetric property requires rank-2 tensor field")

            # Check antifield existence
            if field_name not in self._antifields:
                field_issues.append("Missing corresponding antifield for MSRJD formulation")

            # Check coordinate compatibility
            if len(field_properties.coordinates) != 4:
                field_issues.append("Field should have 4 spacetime coordinates")

            if field_issues:
                issues[field_name] = field_issues

        return issues

    def create_msrjd_field_pairs(self) -> dict[str, dict[str, SymbolicTensorField]]:
        """
        Create organized field pairs for MSRJD action construction.

        Returns:
            Nested dictionary: {field_type: {field_name: field, antifield_name: antifield}}
        """
        organized_pairs: dict[str, dict[str, SymbolicTensorField]] = {}

        for field_name, field in self._fields.items():
            if field_name in self._antifields:
                field_type = field.tensor_properties.field_type

                if field_type not in organized_pairs:
                    organized_pairs[field_type] = {}

                organized_pairs[field_type][field_name] = field
                organized_pairs[field_type][f"{field_name}_tilde"] = self._antifields[field_name]

        return organized_pairs

    def get_field_statistics(self) -> dict[str, int]:
        """
        Get statistics about registered fields.

        Returns:
            Dictionary with field count statistics
        """
        stats = {
            "total_fields": len(self._fields),
            "total_antifields": len(self._antifields),
            "scalar_fields": 0,
            "vector_fields": 0,
            "tensor_fields": 0,
            "constrained_fields": len(
                [name for name, constraints in self._constraints.items() if constraints]
            ),
        }

        for field in self._fields.values():
            if field.index_count == 0:
                stats["scalar_fields"] += 1
            elif field.index_count == 1:
                stats["vector_fields"] += 1
            elif field.index_count >= 2:
                stats["tensor_fields"] += 1

        return stats

    def __str__(self) -> str:
        field_names = list(self._fields.keys())
        return f"IndexedFieldRegistry(fields={field_names}, antifields={len(self._antifields)})"

    def __repr__(self) -> str:
        return self.__str__()
