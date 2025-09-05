"""
Integration tests for SymbolicTensorField bug fix with tensor_msrjd_action.py usage.

This module tests that the create_component bug fix integrates correctly with
the existing usage patterns in the tensor MSRJD action implementation.

Key Integration Points:
    - TensorMSRJDAction usage of create_component method
    - Spatial projector construction with tensor field components
    - Proper tensor component creation in projector calculations
    - Consistency between symbolic tensor and tensor infrastructure
    - Field component creation for physical field calculations

Critical Bug Verification:
    - Ensures create_component produces proper tensor components
    - Verifies spatial projector uses field components correctly
    - Tests integration with existing tensor algebra infrastructure
    - Validates tensor field components work in mathematical expressions
"""

import pytest
import sympy as sp
from sympy import symbols

from rtrg.core.constants import PhysicalConstants
from rtrg.core.tensors import Metric, TensorIndex, TensorIndexStructure
from rtrg.field_theory.symbolic_tensors import (
    IndexedFieldRegistry,
    SymbolicTensorField,
)


class TestTensorMSRJDActionIntegration:
    """Test integration with tensor MSRJD action usage patterns."""

    @pytest.fixture
    def coordinates(self):
        """Standard spacetime coordinates."""
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def metric(self):
        """Minkowski metric for testing."""
        return Metric()

    @pytest.fixture
    def vector_field(self, coordinates):
        """Create four-velocity field like in tensor_msrjd_action.py."""
        return SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)

    def test_create_component_mimics_msrjd_usage(self, vector_field, coordinates):
        """Test the exact usage pattern from tensor_msrjd_action.py."""
        mu, nu = symbols("mu nu")

        # This replicates the lines:
        # u_mu = u_field.create_component([mu], self.coordinates)
        # u_nu = u_field.create_component([nu], self.coordinates)
        u_mu = vector_field.create_component([mu], coordinates)
        u_nu = vector_field.create_component([nu], coordinates)

        # Components should be properly formed tensor components
        u_mu_str = str(u_mu)
        u_nu_str = str(u_nu)

        # Should contain proper component names
        assert "u_mu" in u_mu_str
        assert "u_nu" in u_nu_str

        # Should contain coordinates as function arguments
        for coord in coordinates:
            assert str(coord) in u_mu_str
            assert str(coord) in u_nu_str

    def test_spatial_projector_construction(self, vector_field, coordinates, metric):
        """Test spatial projector construction with tensor field components."""
        mu, nu = symbols("mu nu")

        # Create velocity components like in tensor_msrjd_action.py
        u_mu = vector_field.create_component([mu], coordinates)
        u_nu = vector_field.create_component([nu], coordinates)

        # Create metric components using real Minkowski metric
        # In real code: g_metric = self._get_minkowski_metric()
        # Here we use the actual Minkowski metric tensor
        g_metric_tensor = metric.g  # NumPy array with Minkowski signature (-1, 1, 1, 1)

        # For symbolic computation, we need to create symbolic representation
        # This represents the metric tensor components g^{μν}
        g_metric = sp.IndexedBase("g")

        # Construct spatial projector: g^{μν} + u^μu^ν/c²
        # This replicates: spatial_projector = g_metric[mu, nu] + u_mu * u_nu / PhysicalConstants.c**2
        spatial_projector = g_metric[mu, nu] + u_mu * u_nu / PhysicalConstants.c**2

        # The spatial projector should be a valid SymPy expression
        assert isinstance(spatial_projector, sp.Expr)

        # Should contain the proper tensor components
        projector_str = str(spatial_projector)
        assert "u_mu" in projector_str
        assert "u_nu" in projector_str
        assert str(PhysicalConstants.c) in projector_str

        # Verify the metric tensor is properly created
        assert metric.dim == 4  # 4D spacetime
        assert metric.signature == (-1, 1, 1, 1)  # Minkowski signature
        assert g_metric_tensor.shape == (4, 4)  # 4x4 metric tensor

    def test_component_mathematical_operations(self, vector_field, coordinates):
        """Test that tensor components work in mathematical expressions."""
        mu, nu = symbols("mu nu")

        u_mu = vector_field.create_component([mu], coordinates)
        u_nu = vector_field.create_component([nu], coordinates)

        # Test various mathematical operations
        # Addition
        sum_expr = u_mu + u_nu
        assert isinstance(sum_expr, sp.Expr)

        # Multiplication
        product_expr = u_mu * u_nu
        assert isinstance(product_expr, sp.Expr)

        # Division by constant
        divided_expr = u_mu / PhysicalConstants.c**2
        assert isinstance(divided_expr, sp.Expr)

        # These should all contain proper component names
        for expr in [sum_expr, product_expr, divided_expr]:
            expr_str = str(expr)
            assert any(comp in expr_str for comp in ["u_mu", "u_nu"])

    def test_multiple_field_components_interaction(self, coordinates):
        """Test interaction between components from different tensor fields."""
        # Create multiple fields like in full MSRJD action
        u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        pi_field = SymbolicTensorField(
            "pi", [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")], coordinates
        )

        mu, nu = symbols("mu nu")

        # Create components from different fields
        u_mu = u_field.create_component([mu], coordinates)
        pi_mu_nu = pi_field.create_component([mu, nu], coordinates)

        # Test interaction in expressions
        interaction_expr = u_mu * pi_mu_nu

        interaction_str = str(interaction_expr)
        assert "u_mu" in interaction_str
        assert "pi_mu_nu" in interaction_str

    def test_component_derivatives_work(self, vector_field, coordinates):
        """Test that tensor components can be differentiated properly."""
        mu = symbols("mu")
        t, x, y, z = coordinates

        # Create component
        u_mu = vector_field.create_component([mu], coordinates)

        # Take derivatives with respect to coordinates
        dt_u_mu = sp.diff(u_mu, t)
        dx_u_mu = sp.diff(u_mu, x)

        # Derivatives should be valid expressions
        assert isinstance(dt_u_mu, sp.Expr)
        assert isinstance(dx_u_mu, sp.Expr)

        # Should contain derivative notation
        dt_str = str(dt_u_mu)
        dx_str = str(dx_u_mu)

        # Should show derivatives of u_mu components
        assert "u_mu" in dt_str
        assert "u_mu" in dx_str


class TestSymbolicTensorWithTensorInfrastructure:
    """Test integration with existing tensor infrastructure."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def registry(self):
        return IndexedFieldRegistry()

    def test_registry_integration(self, registry, coordinates):
        """Test IndexedFieldRegistry integration with fixed create_component."""
        # Create Israel-Stewart fields
        registry.create_israel_stewart_fields(coordinates)

        # Get four-velocity field
        u_field = registry.get_field("u")
        assert u_field is not None

        mu = symbols("mu")

        # Create component - this should work with the fix
        u_component = u_field.create_component([mu], coordinates)

        # Should be a proper tensor component
        component_str = str(u_component)
        assert "u_mu" in component_str

    def test_field_action_pairs_with_components(self, registry, coordinates):
        """Test field-antifield pairs can create components properly."""
        registry.create_israel_stewart_fields(coordinates)

        # Get field pairs
        pairs = registry.generate_field_action_pairs()

        mu = symbols("mu")

        for physical_field, response_field in pairs:
            if physical_field.index_count == 1:  # Vector fields
                # Create components from both physical and response fields
                phys_comp = physical_field.create_component([mu], coordinates)
                resp_comp = response_field.create_component([mu], coordinates)

                # Both should be valid expressions
                assert isinstance(phys_comp, sp.Expr)
                assert isinstance(resp_comp, sp.Expr)

                # Should have proper field names
                phys_str = str(phys_comp)
                resp_str = str(resp_comp)

                # Physical field should have base name
                assert physical_field.field_name in phys_str
                # Response field should have _tilde suffix
                assert "_tilde" in resp_str


class TestBugRegressionIntegration:
    """Integration tests specifically targeting the original bug scenario."""

    @pytest.fixture
    def coordinates(self):
        return [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    def test_bug_scenario_spatial_projector(self, coordinates):
        """Test the exact scenario that would have been broken by the original bug."""
        # Set up the exact scenario from tensor_msrjd_action.py
        u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        mu, nu = symbols("mu nu")

        # This is the exact pattern that was failing
        u_mu = u_field.create_component([mu], coordinates)
        u_nu = u_field.create_component([nu], coordinates)

        # Before the fix, these would have been u(mu, t, x, y, z) and u(nu, t, x, y, z)
        # After the fix, they should be u_mu(t, x, y, z) and u_nu(t, x, y, z)

        u_mu_str = str(u_mu)
        u_nu_str = str(u_nu)

        # Critical: should NOT contain the buggy pattern u(mu, ...)
        assert "u(mu," not in u_mu_str
        assert "u(nu," not in u_nu_str

        # Should contain the correct pattern u_mu(...) and u_nu(...)
        assert "u_mu(" in u_mu_str
        assert "u_nu(" in u_nu_str

        # Should contain coordinates as function arguments, not as part of field name
        for coord in coordinates:
            coord_str = str(coord)
            assert coord_str in u_mu_str
            assert coord_str in u_nu_str

    def test_original_vs_fixed_behavior(self, coordinates):
        """Test that demonstrates the difference between buggy and fixed behavior."""
        u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        mu = symbols("mu")

        # Fixed behavior: create_component should produce same result as __getitem__
        component_via_method = u_field.create_component([mu], coordinates)
        component_via_getitem = u_field[mu, *coordinates]

        # Both should produce the same string representation
        method_str = str(component_via_method)
        getitem_str = str(component_via_getitem)

        assert method_str == getitem_str

        # Both should use the proper index encoding
        assert "u_mu" in method_str
        assert "u_mu" in getitem_str

    def test_field_component_in_complex_expression(self, coordinates):
        """Test field components work correctly in complex mathematical expressions."""
        # Create multiple fields
        u_field = SymbolicTensorField("u", [("mu", "upper", "spacetime")], coordinates)
        pi_field = SymbolicTensorField(
            "pi", [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")], coordinates
        )

        mu, nu = symbols("mu nu")

        # Create components
        u_mu = u_field.create_component([mu], coordinates)
        u_nu = u_field.create_component([nu], coordinates)
        pi_mu_nu = pi_field.create_component([mu, nu], coordinates)

        # Create a complex expression like what might appear in MSRJD action
        # Example: (u^μ u^ν + π^{μν}/c²) / (1 + Π)
        complex_expr = u_mu * u_nu + pi_mu_nu / PhysicalConstants.c**2

        # Should be a valid expression
        assert isinstance(complex_expr, sp.Expr)

        # Should contain all the proper component names
        expr_str = str(complex_expr)
        assert "u_mu" in expr_str
        assert "u_nu" in expr_str
        assert "pi_mu_nu" in expr_str

        # Should not contain the buggy patterns
        assert "u(mu" not in expr_str
        assert "u(nu" not in expr_str
        assert "pi(mu" not in expr_str
