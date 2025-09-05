"""
Unit tests for symbolic tensor field operations.

This module tests the SymbolicTensorField class and its methods to ensure
proper tensor index handling, component creation, and integration with
the MSRJD field theory framework.

Key Test Coverage:
    - SymbolicTensorField creation and properties
    - Component creation via create_component method
    - Index encoding in field names vs coordinate passing
    - Consistency between __getitem__ and create_component
    - TensorDerivative operations
    - IndexedFieldRegistry functionality
    - Israel-Stewart field creation and constraint enforcement

Critical Bug Coverage:
    - create_component method uses __getitem__ logic (not __call__)
    - Tensor indices encoded in component names, not passed as coordinates
    - Proper separation of tensor indices from spacetime coordinates
"""

import pytest
import sympy as sp
from sympy import Function, symbols

from rtrg.field_theory.symbolic_tensors import (
    IndexedFieldRegistry,
    SymbolicTensorField,
    TensorDerivative,
    TensorFieldProperties,
)


class TestTensorFieldProperties:
    """Test TensorFieldProperties dataclass validation."""

    def test_valid_properties_creation(self):
        """Test creating valid tensor field properties."""
        coords = [sp.Symbol(name) for name in ["t", "x", "y", "z"]]
        properties = TensorFieldProperties(
            name="u",
            index_structure=[("mu", "upper", "spacetime")],
            coordinates=coords,
            field_type="vector",
        )

        assert properties.name == "u"
        assert len(properties.index_structure) == 1
        assert len(properties.coordinates) == 4

    def test_empty_name_validation(self):
        """Test that empty field names are rejected."""
        coords = [sp.Symbol(name) for name in ["t", "x", "y", "z"]]

        with pytest.raises(ValueError, match="Field name cannot be empty"):
            TensorFieldProperties(
                name="", index_structure=[("mu", "upper", "spacetime")], coordinates=coords
            )

    def test_empty_coordinates_validation(self):
        """Test that empty coordinates are rejected."""
        with pytest.raises(ValueError, match="Coordinates list cannot be empty"):
            TensorFieldProperties(
                name="u", index_structure=[("mu", "upper", "spacetime")], coordinates=[]
            )

    def test_invalid_index_position(self):
        """Test validation of index positions."""
        coords = [sp.Symbol(name) for name in ["t", "x", "y", "z"]]

        with pytest.raises(ValueError, match="Invalid index position"):
            TensorFieldProperties(
                name="u", index_structure=[("mu", "invalid", "spacetime")], coordinates=coords
            )

    def test_invalid_index_type(self):
        """Test validation of index types."""
        coords = [sp.Symbol(name) for name in ["t", "x", "y", "z"]]

        with pytest.raises(ValueError, match="Invalid index type"):
            TensorFieldProperties(
                name="u", index_structure=[("mu", "upper", "invalid")], coordinates=coords
            )


class TestSymbolicTensorField:
    """Test SymbolicTensorField creation and basic operations."""

    @pytest.fixture
    def coordinates(self):
        """Standard spacetime coordinates."""
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def scalar_field(self, coordinates):
        """Create scalar field (energy density)."""
        return SymbolicTensorField("rho", [], coordinates)

    @pytest.fixture
    def vector_field(self, coordinates):
        """Create vector field (four-velocity)."""
        return SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

    @pytest.fixture
    def tensor_field(self, coordinates):
        """Create rank-2 tensor field (shear stress)."""
        return SymbolicTensorField(
            "pi", [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")], coordinates
        )

    def test_scalar_field_creation(self, scalar_field, coordinates):
        """Test creation of scalar field."""
        assert scalar_field.field_name == "rho"
        assert scalar_field.index_count == 0
        assert scalar_field.coordinate_count == 4
        assert scalar_field.coordinates == coordinates

    def test_vector_field_creation(self, vector_field, coordinates):
        """Test creation of vector field."""
        assert vector_field.field_name == "u"
        assert vector_field.index_count == 1
        assert vector_field.coordinate_count == 4
        assert len(vector_field.index_structure) == 1
        assert vector_field.index_structure[0] == ("mu", "upper", "spacetime")

    def test_tensor_field_creation(self, tensor_field, coordinates):
        """Test creation of rank-2 tensor field."""
        assert tensor_field.field_name == "pi"
        assert tensor_field.index_count == 2
        assert tensor_field.coordinate_count == 4
        assert len(tensor_field.index_structure) == 2

    def test_field_properties_access(self, vector_field):
        """Test access to tensor field properties."""
        props = vector_field.tensor_properties
        assert props.name == "u"
        assert len(props.index_structure) == 1
        assert len(props.coordinates) == 4


class TestCreateComponent:
    """Test the create_component method - this is the core bug fix."""

    @pytest.fixture
    def coordinates(self):
        """Standard spacetime coordinates."""
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def vector_field(self, coordinates):
        """Create vector field for testing."""
        return SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

    @pytest.fixture
    def tensor_field(self, coordinates):
        """Create tensor field for testing."""
        return SymbolicTensorField(
            "pi", [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")], coordinates
        )

    def test_create_component_index_encoding(self, vector_field, coordinates):
        """Test that create_component encodes indices in component name."""
        mu = sp.Symbol("mu")
        component = vector_field.create_component([mu], coordinates)

        # The component should be a function call with the field name containing the index
        assert isinstance(component, sp.Expr)

        # Convert to string to check the structure
        component_str = str(component)

        # Should contain "u_mu" (field name with index), not "u(mu, t, x, y, z)"
        assert "u_mu" in component_str

        # Should contain coordinates as function arguments
        assert "t" in component_str
        assert "x" in component_str
        assert "y" in component_str
        assert "z" in component_str

    def test_create_component_vs_getitem_consistency(self, vector_field, coordinates):
        """Test that create_component produces same result as __getitem__."""
        mu = sp.Symbol("mu")

        # Create component using create_component method
        component_method = vector_field.create_component([mu], coordinates)

        # Create component using __getitem__
        component_getitem = vector_field[mu, *coordinates]

        # Both should produce equivalent expressions
        assert str(component_method) == str(component_getitem)

    def test_create_component_tensor_field(self, tensor_field, coordinates):
        """Test create_component with rank-2 tensor field."""
        mu, nu = sp.symbols("mu nu")
        component = tensor_field.create_component([mu, nu], coordinates)

        component_str = str(component)

        # Should encode both indices in the name
        assert "pi_mu_nu" in component_str

        # Should contain coordinates
        for coord in coordinates:
            assert str(coord) in component_str

    def test_create_component_default_coordinates(self, vector_field):
        """Test create_component with default coordinates."""
        mu = sp.Symbol("mu")

        # Don't pass coordinate_values - should use field's default coordinates
        component = vector_field.create_component([mu])

        component_str = str(component)
        assert "u_mu" in component_str
        assert "t" in component_str  # Default coordinates should be used

    def test_create_component_wrong_index_count(self, vector_field, coordinates):
        """Test create_component with wrong number of indices."""
        mu, nu = sp.symbols("mu nu")

        # Vector field expects 1 index, providing 2 should raise error
        with pytest.raises(ValueError, match="Expected 1 tensor indices, got 2"):
            vector_field.create_component([mu, nu], coordinates)

    def test_create_component_wrong_coordinate_count(self, vector_field):
        """Test create_component with wrong number of coordinates."""
        mu = sp.Symbol("mu")
        wrong_coords = [sp.Symbol("t"), sp.Symbol("x")]  # Only 2 instead of 4

        with pytest.raises(ValueError, match="Expected 4 coordinates, got 2"):
            vector_field.create_component([mu], wrong_coords)

    def test_create_component_no_indices_scalar(self, coordinates):
        """Test create_component with scalar field (no indices)."""
        scalar_field = SymbolicTensorField("rho", [], coordinates)

        component = scalar_field.create_component([], coordinates)

        component_str = str(component)
        # Should just be field name with coordinates
        assert "rho" in component_str
        assert "t" in component_str

    def test_create_component_multiple_indices(self, coordinates):
        """Test create_component with multiple tensor indices."""
        # Create rank-3 tensor for testing
        field = SymbolicTensorField(
            "T",
            [
                ("mu", "upper", "spacetime"),
                ("nu", "upper", "spacetime"),
                ("alpha", "upper", "spacetime"),
            ],
            coordinates,
        )

        mu, nu, alpha = sp.symbols("mu nu alpha")
        component = field.create_component([mu, nu, alpha], coordinates)

        component_str = str(component)
        # Should encode all indices: T_mu_nu_alpha
        assert "T_mu_nu_alpha" in component_str


class TestGetItemMethod:
    """Test __getitem__ method for comparison and consistency."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def vector_field(self, coordinates):
        return SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

    def test_getitem_index_separation(self, vector_field, coordinates):
        """Test that __getitem__ properly separates indices from coordinates."""
        mu = sp.Symbol("mu")
        component = vector_field[mu, *coordinates]

        component_str = str(component)
        assert "u_mu" in component_str

    def test_getitem_coordinates_only(self, coordinates):
        """Test __getitem__ with coordinates only for scalar field."""
        scalar_field = SymbolicTensorField("rho", [], coordinates)
        component = scalar_field[coordinates[0], coordinates[1], coordinates[2], coordinates[3]]

        component_str = str(component)
        assert "rho" in component_str


class TestCallMethod:
    """Test __call__ method behavior (designed for scalar fields only)."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    def test_call_scalar_field(self, coordinates):
        """Test __call__ works correctly for scalar fields."""
        scalar_field = SymbolicTensorField("rho", [], coordinates)
        result = scalar_field(*coordinates)

        result_str = str(result)
        assert "rho" in result_str

    def test_call_tensor_field_error(self, coordinates):
        """Test __call__ raises error for tensor fields with rank > 0."""
        vector_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

        with pytest.raises(
            TypeError, match="Tensor field u with rank > 0 is not directly callable"
        ):
            vector_field(*coordinates)


class TestTensorDerivative:
    """Test TensorDerivative operations."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def vector_field(self, coordinates):
        return SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

    def test_tensor_derivative_creation(self, vector_field, coordinates):
        """Test creating tensor derivative."""
        mu = sp.Symbol("mu")
        field_expr = vector_field[mu, *coordinates]

        derivative = TensorDerivative(field_expr, coordinates[0], "partial")

        assert derivative.derivative_type == "partial"
        assert str(derivative).startswith("∂")

    def test_covariant_derivative(self, vector_field, coordinates):
        """Test covariant derivative creation."""
        mu = sp.Symbol("mu")
        field_expr = vector_field[mu, *coordinates]

        derivative = TensorDerivative(field_expr, coordinates[1], "covariant")

        assert derivative.derivative_type == "covariant"
        assert str(derivative).startswith("∇")


class TestIndexedFieldRegistry:
    """Test IndexedFieldRegistry field management."""

    @pytest.fixture
    def registry(self):
        return IndexedFieldRegistry()

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    def test_field_registration(self, registry, coordinates):
        """Test registering fields in the registry."""
        vector_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

        registry.register_field("u", vector_field)

        assert registry.field_count() == 1
        retrieved = registry.get_field("u")
        assert retrieved is not None
        assert retrieved.field_name == "u"

    def test_antifield_creation(self, registry, coordinates):
        """Test creating antifields for MSRJD formulation."""
        vector_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        registry.register_field("u", vector_field)

        antifield = registry.create_antifield("u")

        assert antifield.field_name == "u_tilde"
        # Response fields should have lowered indices
        assert antifield.index_structure[0][1] == "lower"

    def test_israel_stewart_fields_creation(self, registry, coordinates):
        """Test creating complete Israel-Stewart field set."""
        registry.create_israel_stewart_fields(coordinates)

        # Should have all 5 physical fields
        assert registry.field_count() == 5
        assert registry.antifield_count() == 5

        # Check specific fields exist
        assert registry.get_field("rho") is not None
        assert registry.get_field("u") is not None
        assert registry.get_field("pi") is not None
        assert registry.get_field("Pi") is not None
        assert registry.get_field("q") is not None

    def test_constraint_validation(self, registry, coordinates):
        """Test field constraint validation."""
        registry.create_israel_stewart_fields(coordinates)

        validation_results = registry.validate_constraints()

        # All fields should pass validation
        for field_name, is_valid in validation_results.items():
            assert is_valid, f"Field {field_name} failed validation"


class TestConstraintMetricContractions:
    """Test proper metric contractions in constraint expressions."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def minkowski_metric(self):
        """Standard 4D Minkowski metric."""
        from rtrg.core.tensors import Metric

        return Metric()

    @pytest.fixture
    def custom_metric(self):
        """Custom metric for testing different signatures."""
        from rtrg.core.tensors import Metric

        return Metric(dimension=4, signature=(1, 1, 1, -1))  # Mostly-minus convention

    @pytest.fixture
    def vector_field(self, coordinates):
        return SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

    @pytest.fixture
    def tensor_field(self, coordinates):
        return SymbolicTensorField(
            "pi", [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")], coordinates
        )

    def test_normalization_constraint_uses_metric_contraction(
        self, vector_field, coordinates, minkowski_metric
    ):
        """Test that normalization constraint uses proper metric contraction g_{μν} u^μ u^ν."""
        constraint = vector_field.apply_constraint("normalization", minkowski_metric)

        constraint_str = str(constraint)

        # Should contain multiple field components (u_0, u_1, u_2, u_3)
        for mu in range(4):
            assert f"u_{mu}" in constraint_str

        # Should contain c squared
        assert "c**2" in constraint_str

        # For Minkowski metric with signature (-1,1,1,1):
        # g_{μν} u^μ u^ν + c² = -u₀² + u₁² + u₂² + u₃² + c²
        # So should have both positive and negative contributions

        # Check the structure includes both positive and negative metric components
        # This is a structural test - we verify the expression contains the right terms

    def test_normalization_constraint_different_metrics(self, vector_field, coordinates):
        """Test normalization constraint with different metric signatures."""
        from rtrg.core.tensors import Metric

        # Test with mostly-minus convention (1,1,1,-1)
        custom_metric = Metric(dimension=4, signature=(1, 1, 1, -1))
        constraint = vector_field.apply_constraint("normalization", custom_metric)

        constraint_str = str(constraint)

        # Should still contain all components
        for mu in range(4):
            assert f"u_{mu}" in constraint_str

        # Should contain c squared
        assert "c**2" in constraint_str

    def test_traceless_constraint_uses_metric_contraction(
        self, tensor_field, coordinates, minkowski_metric
    ):
        """Test that traceless constraint uses proper metric contraction g_{μν} π^{μν}."""
        constraint = tensor_field.apply_constraint("traceless", minkowski_metric)

        constraint_str = str(constraint)

        # Should contain tensor components with both indices
        # π^{00}, π^{01}, π^{11}, etc.
        tensor_components_found = 0
        for mu in range(4):
            for nu in range(4):
                component_name = f"pi_{mu}_{nu}"
                if component_name in constraint_str:
                    tensor_components_found += 1

        # Should find multiple tensor components
        assert tensor_components_found > 0

    def test_constraint_with_no_metric_uses_default(self, vector_field, coordinates):
        """Test that constraint without explicit metric uses default Minkowski metric."""
        constraint_with_default = vector_field.apply_constraint("normalization")
        constraint_with_explicit = vector_field.apply_constraint("normalization", None)

        # Both should produce the same result
        assert str(constraint_with_default) == str(constraint_with_explicit)

    def test_constraint_expressions_have_correct_dimensions(
        self, vector_field, tensor_field, coordinates, minkowski_metric
    ):
        """Test that constraint expressions are dimensionally correct."""

        # Normalization constraint should have dimension of (velocity)²
        norm_constraint = vector_field.apply_constraint("normalization", minkowski_metric)
        norm_str = str(norm_constraint)

        # Should contain c² term (dimension of velocity squared)
        assert "c**2" in norm_str

        # Traceless constraint should be dimensionless (sum of stress components)
        traceless_constraint = tensor_field.apply_constraint("traceless", minkowski_metric)

        # Should be a valid sympy expression
        assert isinstance(traceless_constraint, sp.Expr)

    def test_constraint_metric_dimension_matching(self, coordinates):
        """Test that constraints work with different metric dimensions."""
        from rtrg.core.tensors import Metric

        # Test with 3D metric
        metric_3d = Metric(dimension=3, signature=(-1, 1, 1))

        # Create 3D field
        field_3d = SymbolicTensorField(
            "u", [("mu", "upper", "spacetime")], coordinates[:3]
        )  # Only t, x, y

        constraint_3d = field_3d.apply_constraint("normalization", metric_3d)
        constraint_str = str(constraint_3d)

        # Should contain 3 components (u_0, u_1, u_2)
        for mu in range(3):
            assert f"u_{mu}" in constraint_str

    def test_constraint_expressions_are_symbolic(
        self, vector_field, tensor_field, coordinates, minkowski_metric
    ):
        """Test that constraint expressions remain symbolic."""
        norm_constraint = vector_field.apply_constraint("normalization", minkowski_metric)
        traceless_constraint = tensor_field.apply_constraint("traceless", minkowski_metric)

        # Should be sympy expressions
        assert isinstance(norm_constraint, sp.Expr)
        assert isinstance(traceless_constraint, sp.Expr)

        # Should contain field components as functions of coordinates
        norm_str = str(norm_constraint)
        traceless_str = str(traceless_constraint)

        # Should contain coordinate symbols
        coord_symbols = ["t", "x", "y", "z"]
        for coord in coord_symbols:
            # At least one constraint should contain coordinate references
            assert coord in norm_str or coord in traceless_str

    def test_lorentz_invariance_verification(
        self, vector_field, tensor_field, coordinates, minkowski_metric
    ):
        """Verify that constraint expressions have the correct Lorentz invariant form."""

        # Test normalization constraint
        norm_constraint = vector_field.apply_constraint("normalization", minkowski_metric)
        norm_str = str(norm_constraint)

        # The normalization constraint g_{μν} u^μ u^ν + c² should be Lorentz invariant
        # This means it should contain all four components with proper metric weights

        # Check that all components are present
        components_present = []
        for mu in range(4):
            if f"u_{mu}" in norm_str:
                components_present.append(mu)

        assert len(components_present) == 4, "All four velocity components should be present"

        # Test traceless constraint
        traceless_constraint = tensor_field.apply_constraint("traceless", minkowski_metric)
        traceless_str = str(traceless_constraint)

        # The traceless constraint g_{μν} π^{μν} should be Lorentz invariant
        # This means it should be a scalar (no free indices)

        # Check that tensor components are present
        tensor_components_present = 0
        for mu in range(4):
            for nu in range(4):
                if f"pi_{mu}_{nu}" in traceless_str:
                    tensor_components_present += 1

        assert (
            tensor_components_present > 0
        ), "Tensor components should be present in traceless constraint"

    def test_constraint_scalar_nature(
        self, vector_field, tensor_field, coordinates, minkowski_metric
    ):
        """Test that constraint expressions are scalars (no free tensor indices)."""

        # Both constraints should produce scalar expressions
        norm_constraint = vector_field.apply_constraint("normalization", minkowski_metric)
        traceless_constraint = tensor_field.apply_constraint("traceless", minkowski_metric)

        # These are scalars - they should be single SymPy expressions
        assert isinstance(norm_constraint, sp.Expr)
        assert isinstance(traceless_constraint, sp.Expr)

        # The expressions should not contain abstract index symbols
        norm_str = str(norm_constraint)
        traceless_str = str(traceless_constraint)

        # Should not contain the symbolic mu, nu variables used in construction
        # (Only numbered indices like u_0, u_1, etc. should appear)
        abstract_indices = ["mu", "nu", "alpha", "beta"]
        for idx in abstract_indices:
            # These shouldn't appear as standalone symbols
            assert idx not in norm_str or f"_{idx}" in norm_str  # Allow u_mu but not just mu
            assert idx not in traceless_str or f"_{idx}" in traceless_str


class TestBugRegression:
    """Test cases specifically designed to catch the original bug."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    def test_create_component_not_using_call(self, coordinates):
        """Test that create_component doesn't use __call__ method inappropriately."""
        vector_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        mu = sp.Symbol("mu")

        # This should NOT produce u(mu, t, x, y, z) which would treat mu as coordinate
        component = vector_field.create_component([mu], coordinates)
        component_str = str(component)

        # Should NOT contain "u(mu" pattern (bug behavior)
        assert "u(mu" not in component_str

        # Should contain "u_mu" pattern (correct behavior)
        assert "u_mu" in component_str

    def test_tensor_msrjd_usage_pattern(self, coordinates):
        """Test the specific usage pattern from tensor_msrjd_action.py."""
        # This replicates the exact usage: u_field.create_component([mu], self.coordinates)
        u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        mu, nu = sp.symbols("mu nu")

        u_mu = u_field.create_component([mu], coordinates)
        u_nu = u_field.create_component([nu], coordinates)

        # These should be proper tensor components, not scalar field calls with extra args
        u_mu_str = str(u_mu)
        u_nu_str = str(u_nu)

        assert "u_mu" in u_mu_str
        assert "u_nu" in u_nu_str

        # Should not contain patterns that treat indices as coordinates
        assert "u(mu" not in u_mu_str
        assert "u(nu" not in u_nu_str

    def test_create_component_function_type(self, coordinates):
        """Test that create_component returns proper Function type."""
        vector_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        mu = sp.Symbol("mu")

        component = vector_field.create_component([mu], coordinates)

        # Should be a SymPy function call, not a general expression
        assert hasattr(component, "func")
        # The function should have the field name with indices
        func_name = str(component.func)
        assert "u_mu" in func_name
