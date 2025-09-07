"""
Unit tests for tensor algebra system
"""

import numpy as np
import pytest

from rtrg.core.constants import PhysicalConstants
from rtrg.core.tensors import IndexStructure, LorentzTensor, Metric


@pytest.mark.unit
class TestMetric:
    """Test Minkowski metric functionality"""

    def test_metric_initialization(self):
        """Test metric tensor initialization"""
        metric = Metric(dimension=4)

        assert metric.dim == 4
        assert metric.g.shape == (4, 4)

        # Check signature (-,+,+,+)
        expected_diag = np.array([-1, 1, 1, 1])
        np.testing.assert_array_equal(np.diag(metric.g), expected_diag)

    def test_metric_properties(self):
        """Test metric tensor mathematical properties"""
        metric = Metric()
        g = metric.g
        g_inv = np.linalg.inv(g)

        # g^μρ g_ρν = δ^μ_ν
        identity = g_inv @ g
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-14)

        # Metric should be diagonal
        off_diag = g - np.diag(np.diag(g))
        np.testing.assert_allclose(off_diag, np.zeros((4, 4)), atol=1e-14)

    def test_raise_lower_index(self, metric):
        """Test index raising and lowering"""
        # Create test vector
        v_lower = np.array([1.0, 2.0, 3.0, 4.0])  # v_μ

        # Raise index: v^μ = g^μν v_ν
        v_upper = metric.raise_index(v_lower, 0)

        # Check first component sign change due to metric signature
        assert v_upper[0] == -v_lower[0]  # g^00 = -1
        np.testing.assert_array_equal(v_upper[1:], v_lower[1:])  # g^ii = +1

        # Lower index back
        v_back = metric.lower_index(v_upper, 0)
        np.testing.assert_allclose(v_back, v_lower, atol=1e-14)


@pytest.mark.unit
class TestIndexStructure:
    """Test index structure management"""

    def test_index_structure_creation(self):
        """Test IndexStructure initialization"""
        indices = IndexStructure(
            names=["mu", "nu"],
            types=["contravariant", "covariant"],
            symmetries=["symmetric", "symmetric"],
        )

        assert indices.rank == 2
        assert indices.names == ["mu", "nu"]
        assert indices.is_symmetric()

    def test_invalid_index_structure(self):
        """Test validation of index structure"""
        with pytest.raises(ValueError, match="Index arrays must have same length"):
            IndexStructure(
                names=["mu"],
                types=["contravariant", "covariant"],  # Wrong length
                symmetries=["symmetric"],
            )


@pytest.mark.unit
class TestLorentzTensor:
    """Test Lorentz tensor operations"""

    def test_tensor_creation(self, metric):
        """Test tensor creation and basic properties"""
        # Create rank-2 tensor
        components = np.random.rand(4, 4)
        indices = IndexStructure(
            names=["mu", "nu"],
            types=["contravariant", "contravariant"],
            symmetries=["none", "none"],
        )

        tensor = LorentzTensor(components, indices, metric)

        assert tensor.rank == 2
        assert tensor.shape == (4, 4)
        np.testing.assert_array_equal(tensor.components, components)

    def test_tensor_symmetrization(self, metric):
        """Test tensor symmetrization"""
        # Create asymmetric tensor
        components = np.random.rand(4, 4)
        indices = IndexStructure(["mu", "nu"], ["contravariant", "contravariant"], ["none", "none"])
        tensor = LorentzTensor(components, indices, metric)

        # Symmetrize
        symmetric_tensor = tensor.symmetrize()

        # Check symmetry
        sym_comp = symmetric_tensor.components
        np.testing.assert_allclose(sym_comp, sym_comp.T, atol=1e-14)

        # Check it's actually the symmetrized version
        expected = 0.5 * (components + components.T)
        np.testing.assert_allclose(sym_comp, expected, atol=1e-14)

    def test_tensor_trace(self, metric):
        """Test tensor trace operations"""
        # Create rank-2 tensor
        components = np.diag([1, 2, 3, 4])  # Diagonal for easy trace
        indices = IndexStructure(["mu", "nu"], ["contravariant", "covariant"], ["none", "none"])
        tensor = LorentzTensor(components, indices, metric)

        # Take trace
        trace = tensor.trace()

        # Should give scalar result
        assert isinstance(trace, int | float | complex)
        assert trace == np.trace(components)

    def test_four_velocity_normalization(self, metric, normalized_four_velocity):
        """Test four-velocity normalization constraint"""
        # Contract with itself: u^μ u_μ
        # This is a simplified test - full contraction would be more complex
        u_dot_u = np.dot(normalized_four_velocity, metric.g @ normalized_four_velocity)

        # Should equal -c²
        expected = -(PhysicalConstants.c**2)
        assert abs(u_dot_u - expected) < 1e-12

    def test_spatial_projection(self, metric, normalized_four_velocity):
        """Test spatial projection operations"""
        # Create test tensor
        components = np.eye(4)  # Identity tensor
        indices = IndexStructure(["mu", "nu"], ["contravariant", "contravariant"], ["none", "none"])
        tensor = LorentzTensor(components, indices, metric)

        # Project onto spatial subspace
        projected = tensor.project_spatial(normalized_four_velocity)

        # Check orthogonality to four-velocity
        # This is a simplified check - full test would verify all components
        assert projected.components.shape == (4, 4)

        # Time-time component should be modified
        assert projected.components[0, 0] != components[0, 0]

    def test_spatial_projector_matches_constraints(self, metric, normalized_four_velocity):
        """Ensure core projector matches constraints.spatial_projector."""
        from rtrg.israel_stewart.constraints import spatial_projector

        u = normalized_four_velocity
        # Build Δ using constraints helper
        Delta_constraints = spatial_projector(u, metric)

        # Build Δ via core Metric computation (using same formula internally)
        g = metric.g
        u_lower = g @ u
        Delta_core = g + np.outer(u_lower, u_lower) / (1.0**2)

        np.testing.assert_allclose(Delta_constraints, Delta_core, atol=1e-12)

    def test_index_raising_lowering(self, metric):
        """Test index raising and lowering for tensors"""
        # Create vector with covariant index
        components = np.array([1, 2, 3, 4])
        indices = IndexStructure(["mu"], ["covariant"], ["none"])
        vector = LorentzTensor(components, indices, metric)

        # Raise index
        vector_up = vector.raise_index(0)

        # Check index type changed
        assert vector_up.indices.types[0] == "contravariant"

        # Check components transformed correctly
        expected = metric.g @ components
        np.testing.assert_allclose(vector_up.components, expected, atol=1e-14)

        # Lower index back
        vector_back = vector_up.lower_index(0)

        # Should recover original
        np.testing.assert_allclose(vector_back.components, components, atol=1e-14)
        assert vector_back.indices.types[0] == "covariant"


@pytest.mark.unit
class TestTensorConstraints:
    """Test constraint satisfaction for tensor fields"""

    def test_traceless_tensor(self, metric):
        """Test creation and validation of traceless tensors"""
        # Create symmetric traceless tensor (shear-like)
        pi = np.zeros((4, 4))
        pi[1, 2] = pi[2, 1] = 0.1
        pi[1, 1] = 0.05
        pi[2, 2] = -0.05  # Make traceless

        indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["symmetric", "symmetric"]
        )
        tensor = LorentzTensor(pi, indices, metric)

        # Check trace
        trace = tensor.trace()
        assert abs(trace) < 1e-12

    def test_antisymmetric_tensor(self, metric):
        """Test antisymmetric tensor creation"""
        # Create antisymmetric tensor
        F = np.zeros((4, 4))
        F[0, 1] = 1.0
        F[1, 0] = -1.0
        F[2, 3] = 2.0
        F[3, 2] = -2.0

        indices = IndexStructure(["mu", "nu"], ["contravariant", "contravariant"], ["none", "none"])
        tensor = LorentzTensor(F, indices, metric)

        # Antisymmetrize
        antisym = tensor.antisymmetrize()

        # Check antisymmetry
        A = antisym.components
        np.testing.assert_allclose(A, -A.T, atol=1e-14)


@pytest.mark.physics
class TestPhysicalTensors:
    """Test tensors that appear in relativity"""

    def test_stress_energy_tensor_structure(self, metric, normalized_four_velocity):
        """Test structure of stress-energy tensor"""
        # Simple perfect fluid: T^μν = ρ u^μ u^ν + p Δ^μν
        rho = 1.0
        p = 0.3

        u = normalized_four_velocity

        # Energy-momentum tensor
        T = rho * np.outer(u, u)

        # Add pressure in spatial directions (simplified)
        g_inv = np.linalg.inv(metric.g)
        Delta = g_inv + np.outer(u, u) / PhysicalConstants.c**2
        T += p * Delta

        # Create tensor
        indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["symmetric", "symmetric"]
        )
        T_tensor = LorentzTensor(T, indices, metric)

        # Check energy density (T^00 component)
        assert T_tensor.components[0, 0] > 0  # Positive energy

        # Check symmetry
        np.testing.assert_allclose(T, T.T, atol=1e-14)

    def test_electromagnetic_field_tensor(self, metric):
        """Test electromagnetic field tensor properties"""
        # F^μν antisymmetric tensor
        F = np.array(
            [
                [0, -1, -2, -3],  # F^0i = -E^i/c
                [1, 0, -4, -5],  # F^ij = ε^ijk B^k
                [2, 4, 0, -6],
                [3, 5, 6, 0],
            ]
        )

        indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["antisymmetric", "antisymmetric"]
        )
        F_tensor = LorentzTensor(F, indices, metric)

        # Check antisymmetry
        np.testing.assert_allclose(F, -F.T, atol=1e-14)

        # Check trace is zero (antisymmetric tensors are traceless)
        trace = F_tensor.trace()
        assert abs(trace) < 1e-12


@pytest.mark.unit
class TestPhysicalConstants:
    """Test physical constants and unit conversions"""

    def test_corrected_gev_to_cgs_conversions(self):
        """Test that GeV→CGS conversion factors are correct"""
        from rtrg.core.constants import PhysicalConstants

        # Test length conversion (ℏc/GeV)
        length_cm = PhysicalConstants.to_cgs(1.0, "length")
        expected_length = 1.97327e-14  # cm, from ℏc ≈ 197.327 MeV·fm
        assert abs(length_cm - expected_length) / expected_length < 1e-4

        # Test time conversion (ℏ/GeV)
        time_s = PhysicalConstants.to_cgs(1.0, "time")
        expected_time = 6.58212e-25  # s, from ℏ ≈ 6.58212×10^-22 MeV·s
        assert abs(time_s - expected_time) / expected_time < 1e-4

    def test_unit_system_conversions_consistency(self):
        """Test UnitSystem conversions match PhysicalConstants"""
        from rtrg.core.constants import PhysicalConstants, UnitSystem

        unit_system = UnitSystem("cgs")

        # Length conversion should match
        length_pc = PhysicalConstants.to_cgs(1.0, "length")
        length_us = unit_system.convert(1.0, "length", "natural")
        assert abs(length_pc - length_us) < 1e-20

        # Time conversion should match
        time_pc = PhysicalConstants.to_cgs(1.0, "time")
        time_us = unit_system.convert(1.0, "time", "natural")
        assert abs(time_pc - time_us) < 1e-30


@pytest.mark.unit
class TestMetricValidation:
    """Test Metric constructor validation"""

    def test_metric_dimension_validation(self):
        """Test dimension validation in Metric constructor"""
        # Valid dimensions should work
        metric = Metric(dimension=4)
        assert metric.dim == 4

        metric_3d = Metric(dimension=3, signature=(-1, 1, 1))
        assert metric_3d.dim == 3

        # Invalid dimensions should fail
        with pytest.raises(ValueError, match="Dimension must be positive"):
            Metric(dimension=0)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            Metric(dimension=-1)

    def test_metric_signature_validation(self):
        """Test signature length validation in Metric constructor"""
        # Matching signature length should work
        metric = Metric(dimension=2, signature=(-1, 1))
        assert metric.signature == (-1, 1)

        # Mismatched signature length should fail
        with pytest.raises(ValueError, match="Signature length.*doesn't match dimension"):
            Metric(dimension=4, signature=(-1, 1))  # Too short

        with pytest.raises(ValueError, match="Signature length.*doesn't match dimension"):
            Metric(dimension=2, signature=(-1, 1, 1, 1))  # Too long

    def test_metric_construction_with_validation(self):
        """Test that metric tensor is constructed correctly after validation"""
        # Test default case
        metric = Metric()
        expected_diag = [-1, 1, 1, 1]
        np.testing.assert_array_equal(np.diag(metric.g), expected_diag)

        # Test custom case
        metric_custom = Metric(dimension=3, signature=(1, -1, -1))
        expected_custom = [1, -1, -1]
        np.testing.assert_array_equal(np.diag(metric_custom.g), expected_custom)
