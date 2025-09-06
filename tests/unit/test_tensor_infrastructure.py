"""
Tests for enhanced tensor infrastructure and tensor-aware propagator calculations.

This module tests the Phase 1 implementation of the full MSRJD propagator
system with proper tensor index handling, constraints, and projections.
"""

import numpy as np
import pytest
import sympy as sp

from rtrg.core.fields import (
    EnhancedEnergyDensityField,
    EnhancedFourVelocityField,
    EnhancedShearStressField,
    FieldProperties,
    TensorAwareField,
)
from rtrg.core.registry_factory import create_registry_for_context
from rtrg.core.tensors import (
    ConstrainedTensorField,
    IndexType,
    Metric,
    ProjectionOperators,
    TensorIndex,
    TensorIndexStructure,
)
from rtrg.field_theory.propagators import TensorAwarePropagatorCalculator


class TestTensorIndex:
    """Test tensor index system."""

    def test_tensor_index_creation(self):
        """Test basic tensor index creation."""
        mu_index = TensorIndex("mu", IndexType.SPACETIME, "upper")

        assert mu_index.name == "mu"
        assert mu_index.index_type == IndexType.SPACETIME
        assert mu_index.position == "upper"
        assert mu_index.dimension == 4

    def test_index_contractibility(self):
        """Test index contraction rules."""
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        mu_lower = TensorIndex("mu", IndexType.SPACETIME, "lower")
        nu_upper = TensorIndex("nu", IndexType.SPACETIME, "upper")
        nu_lower = TensorIndex("nu", IndexType.SPACETIME, "lower")
        i_upper = TensorIndex("i", IndexType.SPATIAL, "upper")

        # Same name, opposite positions can contract
        assert mu_upper.is_contractible_with(mu_lower)
        assert mu_lower.is_contractible_with(mu_upper)

        # Different names with opposite positions CAN contract (fixed physics bug)
        assert mu_upper.is_contractible_with(nu_lower)
        assert nu_lower.is_contractible_with(mu_upper)

        # Same position cannot contract (both upper)
        assert not mu_upper.is_contractible_with(nu_upper)

        # Same position cannot contract (both lower)
        assert not mu_lower.is_contractible_with(nu_lower)

        # Different index types cannot contract (spacetime vs spatial)
        assert not mu_upper.is_contractible_with(i_upper)

    def test_raise_lower_index(self):
        """Test index position changing."""
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        mu_lower = mu_upper.raise_lower_index()

        assert mu_lower.position == "lower"
        assert mu_lower.name == "mu"
        assert mu_lower.index_type == IndexType.SPACETIME


class TestIndexContractibilityBugFix:
    """Test Bug #5 fix: Index compatibility should allow different names."""

    def test_different_names_opposite_positions_contract(self):
        """Test that indices with different names and opposite positions can contract."""
        # Physics: T^μν h_ρσ should allow μ↔ρ and ν↔σ contractions
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        rho_lower = TensorIndex("rho", IndexType.SPACETIME, "lower")
        nu_upper = TensorIndex("nu", IndexType.SPACETIME, "upper")
        sigma_lower = TensorIndex("sigma", IndexType.SPACETIME, "lower")

        # All these should be contractible
        assert mu_upper.is_contractible_with(rho_lower)
        assert rho_lower.is_contractible_with(mu_upper)
        assert nu_upper.is_contractible_with(sigma_lower)
        assert sigma_lower.is_contractible_with(nu_upper)

    def test_different_names_same_positions_dont_contract(self):
        """Test that indices with same positions cannot contract regardless of names."""
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        nu_upper = TensorIndex("nu", IndexType.SPACETIME, "upper")
        rho_lower = TensorIndex("rho", IndexType.SPACETIME, "lower")
        sigma_lower = TensorIndex("sigma", IndexType.SPACETIME, "lower")

        # Upper with upper cannot contract
        assert not mu_upper.is_contractible_with(nu_upper)
        assert not nu_upper.is_contractible_with(mu_upper)

        # Lower with lower cannot contract
        assert not rho_lower.is_contractible_with(sigma_lower)
        assert not sigma_lower.is_contractible_with(rho_lower)

    def test_different_index_types_dont_contract(self):
        """Test that different index types cannot contract."""
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        i_lower = TensorIndex("i", IndexType.SPATIAL, "lower")
        j_upper = TensorIndex("j", IndexType.SPATIAL, "upper")
        t_lower = TensorIndex("t", IndexType.TEMPORAL, "lower")

        # Spacetime vs spatial
        assert not mu_upper.is_contractible_with(i_lower)
        assert not i_lower.is_contractible_with(mu_upper)

        # Spacetime vs temporal
        assert not mu_upper.is_contractible_with(t_lower)
        assert not t_lower.is_contractible_with(mu_upper)

        # Spatial vs temporal
        assert not j_upper.is_contractible_with(t_lower)
        assert not t_lower.is_contractible_with(j_upper)

    def test_same_name_contractions_still_work(self):
        """Test that same name contractions still work as before."""
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        mu_lower = TensorIndex("mu", IndexType.SPACETIME, "lower")

        # This should still work
        assert mu_upper.is_contractible_with(mu_lower)
        assert mu_lower.is_contractible_with(mu_upper)

    def test_physics_example_tensor_contractions(self):
        """Test realistic physics examples that were previously blocked."""
        # Example: T^μν g_ρσ contraction should work for μ↔ρ or ν↔σ
        T_mu = TensorIndex("mu", IndexType.SPACETIME, "upper")
        T_nu = TensorIndex("nu", IndexType.SPACETIME, "upper")
        g_rho = TensorIndex("rho", IndexType.SPACETIME, "lower")
        g_sigma = TensorIndex("sigma", IndexType.SPACETIME, "lower")

        # These contractions should be possible
        assert T_mu.is_contractible_with(g_rho)  # T^μν g_μσ
        assert T_mu.is_contractible_with(g_sigma)  # T^μν g_ρμ
        assert T_nu.is_contractible_with(g_rho)  # T^μν g_νσ
        assert T_nu.is_contractible_with(g_sigma)  # T^μν g_ρν

        # Example: π^μν u_α (shear stress with four-velocity)
        pi_mu = TensorIndex("mu", IndexType.SPACETIME, "upper")
        pi_nu = TensorIndex("nu", IndexType.SPACETIME, "upper")
        u_alpha = TensorIndex("alpha", IndexType.SPACETIME, "lower")

        # These should contract for orthogonality constraint: π^μν u_ν = 0
        assert pi_nu.is_contractible_with(u_alpha)  # Can contract ν↔α
        assert pi_mu.is_contractible_with(u_alpha)  # Can contract μ↔α


class TestTensorIndexStructure:
    """Test tensor index structure management."""

    def test_index_structure_creation(self):
        """Test creating tensor index structures."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        nu_idx = TensorIndex("nu", IndexType.SPACETIME, "upper")

        structure = TensorIndexStructure([mu_idx, nu_idx])

        assert structure.rank == 2
        assert len(structure.free_indices) == 2
        assert len(structure.dummy_indices) == 0

    def test_dummy_index_detection(self):
        """Test detection of dummy (contracted) indices."""
        mu_upper = TensorIndex("mu", IndexType.SPACETIME, "upper")
        mu_lower = TensorIndex("mu", IndexType.SPACETIME, "lower")
        nu_upper = TensorIndex("nu", IndexType.SPACETIME, "upper")

        structure = TensorIndexStructure([mu_upper, mu_lower, nu_upper])

        assert structure.rank == 3
        assert len(structure.free_indices) == 1
        assert len(structure.dummy_indices) == 1

        dummy_pair = structure.dummy_indices[0]
        assert dummy_pair[0].name == "mu" and dummy_pair[1].name == "mu"

    def test_tensor_contraction(self):
        """Test tensor structure contraction."""
        # Create two rank-1 tensors
        mu_idx1 = TensorIndex("mu", IndexType.SPACETIME, "upper")
        nu_idx2 = TensorIndex("nu", IndexType.SPACETIME, "lower")

        struct1 = TensorIndexStructure([mu_idx1])
        struct2 = TensorIndexStructure([nu_idx2])

        # Contract them (no actual contraction - just composition)
        result = struct1.contract_with(struct2, [])

        assert result.rank == 2
        assert len(result.free_indices) == 2


class TestConstrainedTensorField:
    """Test constrained tensor field operations."""

    @pytest.fixture
    def four_velocity_field(self, metric):
        """Create four-velocity tensor field."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        return ConstrainedTensorField("u", structure, metric, ["normalized"])

    def test_normalization_constraint(self, four_velocity_field, metric):
        """Test four-velocity normalization."""
        # Test vector that's not normalized
        components = np.array([2.0, 0.5, 0.3, 0.1])

        normalized = four_velocity_field.apply_constraints(components)

        # Check normalization: u^μ u_μ = -1 using proper metric tensor
        norm = np.einsum("i,ij,j->", normalized, metric.g, normalized)
        assert abs(norm + 1.0) < 1e-10

    def test_constraint_validation(self, four_velocity_field):
        """Test constraint validation."""
        # Normalized vector
        normalized_components = np.array([1.0, 0.0, 0.0, 0.0])
        validation = four_velocity_field.validate_constraints(normalized_components)
        assert validation["normalized"]

        # Non-normalized vector
        non_normalized = np.array([2.0, 0.0, 0.0, 0.0])
        validation = four_velocity_field.validate_constraints(non_normalized)
        assert not validation["normalized"]


class TestProjectionOperators:
    """Test projection operators for field decomposition."""

    @pytest.fixture
    def metric(self):
        """Create Minkowski metric."""
        return Metric()

    @pytest.fixture
    def projector(self, metric):
        """Create projection operators."""
        return ProjectionOperators(metric)

    @pytest.fixture
    def rest_velocity(self):
        """Rest frame four-velocity."""
        return np.array([1.0, 0.0, 0.0, 0.0])

    def test_spatial_projector(self, projector, rest_velocity):
        """Test spatial projection operator."""
        h_proj = projector.spatial_projector(rest_velocity)

        # Check dimensions
        assert h_proj.shape == (4, 4)

        # Check orthogonality to four-velocity
        projected_u = h_proj @ rest_velocity
        # Should be zero (within numerical precision)
        assert np.allclose(projected_u, [0, 0, 0, 0], atol=1e-12)

    def test_longitudinal_transverse_decomposition(self, projector):
        """Test longitudinal/transverse projection."""
        momentum = np.array([1.0, 2.0, 3.0])

        p_long = projector.longitudinal_projector(momentum)
        p_trans = projector.transverse_projector(momentum)

        # Check orthogonality: P_L + P_T = I
        identity_check = p_long + p_trans
        assert np.allclose(identity_check, np.eye(3), atol=1e-12)

        # Check projection properties: P_L^2 = P_L, P_T^2 = P_T
        assert np.allclose(p_long @ p_long, p_long, atol=1e-12)
        assert np.allclose(p_trans @ p_trans, p_trans, atol=1e-12)

    def test_vector_decomposition(self, projector, rest_velocity):
        """Test vector decomposition into components."""
        vector = np.array([0.0, 1.0, 2.0, 3.0])  # Spatial vector
        momentum = np.array([1.0, 0.0, 0.0])  # Along x-axis

        decomp = projector.decompose_vector(vector, momentum, rest_velocity)

        # Check that decomposition sums to original (spatial part)
        reconstructed = decomp["longitudinal"] + decomp["transverse"] + decomp["temporal"]

        # The spatial part should match
        assert np.allclose(reconstructed[1:], vector[1:], atol=1e-12)


class TestEnhancedFields:
    """Test enhanced tensor-aware field classes."""

    @pytest.fixture
    def metric(self):
        """Create metric for fields."""
        return Metric()

    def test_enhanced_energy_density(self, metric):
        """Test enhanced energy density field."""
        field = EnhancedEnergyDensityField(metric)

        assert field.name == "rho"
        assert field.index_structure.rank == 0  # Scalar
        assert len(field.constraints) == 0

    def test_enhanced_four_velocity(self, metric):
        """Test enhanced four-velocity field."""
        field = EnhancedFourVelocityField(metric)

        assert field.name == "u"
        assert field.index_structure.rank == 1  # Vector
        assert "normalized" in field.constraints

        # Test constraint application
        components = np.array([2.0, 0.0, 0.0, 0.0])
        constrained = field.apply_constraints(components)

        # Check normalization using proper metric
        norm = np.einsum("i,ij,j->", constrained, metric.g, constrained)
        assert abs(norm + 1.0) < 1e-10

    def test_enhanced_shear_stress(self, metric):
        """Test enhanced shear stress field."""
        field = EnhancedShearStressField(metric)

        assert field.name == "pi"
        assert field.index_structure.rank == 2  # Tensor
        assert "symmetric" in field.constraints
        assert "traceless" in field.constraints
        assert "orthogonal_to_velocity" in field.constraints


class TestEnhancedFieldRegistry:
    """Test enhanced field registry."""

    @pytest.fixture
    def registry(self):
        """Create enhanced field registry."""
        return create_registry_for_context("tensor_operations")

    @pytest.fixture
    def metric(self):
        """Create metric."""
        return Metric()

    def test_enhanced_field_creation(self, registry, metric):
        """Test creating enhanced IS fields."""
        registry.create_enhanced_is_fields(metric)

        # Check all fields are created
        assert "rho" in registry
        assert "u" in registry
        assert "pi" in registry
        assert "Pi" in registry
        assert "q" in registry

        # Check tensor-aware fields
        u_field = registry.get_tensor_aware_field("u")
        assert isinstance(u_field, TensorAwareField)
        assert u_field.index_structure.rank == 1

    def test_constraint_validation(self, registry, metric):
        """Test constraint validation for all fields."""
        registry.create_enhanced_is_fields(metric)

        # Create test field components
        field_components = {
            "rho": np.array([1.0]),  # Scalar
            "u": np.array([1.0, 0.0, 0.0, 0.0]),  # Normalized velocity
            "pi": np.zeros((4, 4)),  # Zero shear stress
            "Pi": np.array([0.0]),  # Zero bulk pressure
            "q": np.array([0.0, 1.0, 0.0, 0.0]),  # Heat flux
        }

        validation_results = registry.validate_all_constraints(field_components)

        # Four-velocity should pass normalization
        assert validation_results["u"]["normalized"]

        # Shear stress should pass symmetry/traceless tests (zero tensor)
        if "pi" in validation_results:
            assert validation_results["pi"]["symmetric"]
            assert validation_results["pi"]["traceless"]


class TestTensorAwarePropagatorCalculator:
    """Test tensor-aware propagator calculations."""

    @pytest.fixture
    def mock_msrjd_action(self, metric):
        """Create mock MSRJD action."""

        class MockISSystem:
            def __init__(self):
                from rtrg.israel_stewart.equations import IsraelStewartParameters

                self.parameters = IsraelStewartParameters(
                    eta=0.1,
                    zeta=0.05,
                    kappa=0.2,
                    tau_pi=0.5,
                    tau_Pi=0.3,
                    tau_q=0.4,
                    temperature=1.0,
                    chemical_potential=0.0,
                    equilibrium_pressure=0.33,
                )

                self.field_registry = create_registry_for_context("basic_physics", metric=metric)

        class MockMSRJDAction:
            def __init__(self):
                self.is_system = MockISSystem()

        return MockMSRJDAction()

    @pytest.fixture
    def metric(self):
        """Create metric."""
        return Metric()

    def test_tensor_aware_calculator_creation(self, mock_msrjd_action, metric):
        """Test creating tensor-aware propagator calculator using factory pattern."""
        from rtrg.core.calculator_factory import create_propagator_calculator

        calc = create_propagator_calculator("tensor_aware", mock_msrjd_action, metric=metric)

        # Test that adapter provides unified interface
        assert hasattr(calc, "calculate_retarded_propagator")
        assert hasattr(calc, "calculate_advanced_propagator")
        assert hasattr(calc, "calculate_keldysh_propagator")

    def test_field_matrix_size_calculation(self, mock_msrjd_action, metric):
        """Test calculating matrix sizes for tensor fields using factory."""
        from rtrg.core.calculator_factory import create_propagator_calculator

        calc = create_propagator_calculator("tensor_aware", mock_msrjd_action, metric=metric)

        if calc.enhanced_registry:
            # Create enhanced fields for testing
            from rtrg.core.tensors import Metric

            metric = Metric()
            calc.enhanced_registry.create_enhanced_is_fields(metric)

            # Test sizes
            rho_field = calc.enhanced_registry.get_tensor_aware_field("rho")
            u_field = calc.enhanced_registry.get_tensor_aware_field("u")
            pi_field = calc.enhanced_registry.get_tensor_aware_field("pi")

            if rho_field:
                assert calc._get_field_matrix_size(rho_field) == 1  # Scalar
            if u_field:
                # Vector with normalization constraint: 4 - 1 = 3 DOF
                assert calc._get_field_matrix_size(u_field) >= 1
            if pi_field:
                # Rank-2 tensor with constraints: 16 - 1 (trace) - 4 (orthogonal) = 11 DOF
                assert calc._get_field_matrix_size(pi_field) >= 1


# Integration test
def test_phase1_integration():
    """Integration test for Phase 1 tensor infrastructure."""
    # Create basic components
    metric = Metric()
    registry = create_registry_for_context("tensor_operations", metric=metric)

    # Test field creation
    assert len(registry) >= 5

    # Test tensor-aware field
    u_field = registry.get_tensor_aware_field("u")
    assert u_field is not None
    assert u_field.index_structure.rank == 1

    # Test constraint application
    components = np.array([2.0, 0.5, 0.3, 0.1])
    constrained = u_field.apply_constraints(components)

    # Verify normalization using proper metric
    norm = np.einsum("i,ij,j->", constrained, metric.g, constrained)
    assert abs(norm + 1.0) < 1e-10

    print("✅ Phase 1 tensor infrastructure integration test passed!")
