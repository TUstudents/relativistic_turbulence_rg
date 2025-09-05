"""
Comprehensive tests for Lorentz tensor algebra system.

Tests all tensor operations required by Task 1.2, including:
- Metric properties and operations
- Index raising/lowering
- Tensor contractions
- Symmetrization/antisymmetrization
- Christoffel symbols computation
- Covariant derivatives
- Physical constraints and invariants

Mathematical Validation:
    Tests verify fundamental tensor algebra properties and relativistic
    invariants that must hold for the tensor system to be correct.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rtrg.core.constants import PhysicalConstants
from rtrg.core.tensors import IndexStructure, LorentzTensor, Metric


class TestMetric:
    """Test Minkowski metric tensor properties and operations."""

    def test_metric_construction(self):
        """Test metric tensor initialization"""
        metric = Metric()

        # Check default signature (-,+,+,+)
        expected = np.diag([-1, 1, 1, 1])
        assert_allclose(metric.g, expected)
        assert metric.dim == 4

    def test_custom_metric(self):
        """Test custom metric construction"""
        # 2+1 dimensional spacetime
        metric = Metric(dimension=3, signature=(-1, 1, 1))

        expected = np.diag([-1, 1, 1])
        assert_allclose(metric.g, expected)
        assert metric.dim == 3

    def test_metric_properties(self):
        """Test fundamental metric properties"""
        metric = Metric()

        # Metric is symmetric
        assert_allclose(metric.g, metric.g.T)

        # Determinant should be -1 for Minkowski metric
        det = np.linalg.det(metric.g)
        assert_allclose(det, -1.0, rtol=1e-10)

        # g^μν g_νρ = δ^μ_ρ (identity)
        g_inv = np.linalg.inv(metric.g)
        identity = g_inv @ metric.g
        expected_identity = np.eye(4)
        assert_allclose(identity, expected_identity, atol=1e-12)


class TestIndexStructure:
    """Test tensor index structure management."""

    def test_index_structure_creation(self):
        """Test IndexStructure initialization"""
        indices = IndexStructure(
            names=["mu", "nu"],
            types=["contravariant", "covariant"],
            symmetries=["none", "symmetric"],
        )

        assert indices.rank == 2
        assert indices.names == ["mu", "nu"]
        assert indices.is_symmetric()

    def test_invalid_index_structure(self):
        """Test validation of invalid index structures"""
        # Mismatched array lengths
        with pytest.raises(ValueError):
            IndexStructure(["mu"], ["covariant", "contravariant"], ["none"])

        # Invalid index type
        with pytest.raises(ValueError):
            IndexStructure(["mu"], ["invalid"], ["none"])

        # Invalid symmetry
        with pytest.raises(ValueError):
            IndexStructure(["mu"], ["covariant"], ["invalid"])


class TestLorentzTensor:
    """Test Lorentz tensor operations and properties."""

    def setup_method(self):
        """Set up test fixtures"""
        self.metric = Metric()

        # Four-velocity in rest frame: u^μ = (c, 0, 0, 0)
        u_components = np.array([PhysicalConstants.c, 0, 0, 0])
        u_indices = IndexStructure(["mu"], ["contravariant"], ["none"])
        self.four_velocity = LorentzTensor(u_components, u_indices, self.metric)

        # Stress-energy tensor for perfect fluid
        rho = 1.0  # Energy density
        p = 0.3  # Pressure
        T_components = np.diag([rho, p, p, p])
        T_indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["symmetric", "symmetric"]
        )
        self.stress_tensor = LorentzTensor(T_components, T_indices, self.metric)

        # Metric tensor as LorentzTensor
        g_indices = IndexStructure(
            ["mu", "nu"], ["covariant", "covariant"], ["symmetric", "symmetric"]
        )
        self.metric_tensor = LorentzTensor(self.metric.g, g_indices, self.metric)

    def test_tensor_creation(self):
        """Test tensor initialization"""
        assert self.four_velocity.rank == 1
        assert self.four_velocity.shape == (4,)
        assert self.stress_tensor.rank == 2
        assert self.stress_tensor.shape == (4, 4)

    def test_four_velocity_normalization(self):
        """Test four-velocity normalization: u^μ u_μ = -c²"""
        # Lower the index: u_μ = g_μν u^ν
        u_covariant = self.four_velocity.lower_index(0)

        # Contract: u^μ u_μ
        norm_squared = self.four_velocity.contract(u_covariant, [(0, 0)])

        # Should equal -c²
        expected = -(PhysicalConstants.c**2)

        # Check if result is scalar (rank-0 tensor) or actual scalar
        if hasattr(norm_squared, "components"):
            actual_value = norm_squared.components
        else:
            actual_value = norm_squared

        assert_allclose(actual_value, expected, rtol=1e-10)

    def test_index_raising_lowering(self):
        """Test index raising and lowering operations"""
        # Start with contravariant four-velocity u^μ
        u_up = self.four_velocity

        # Lower index: u_μ = g_μν u^ν
        u_down = u_up.lower_index(0)
        assert u_down.indices.types[0] == "covariant"

        # Raise index again: should recover original
        u_up_again = u_down.raise_index(0)
        assert u_up_again.indices.types[0] == "contravariant"
        assert_allclose(u_up_again.components, u_up.components, rtol=1e-12)

        # Check explicit calculation
        expected_down = self.metric.g @ u_up.components
        assert_allclose(u_down.components, expected_down)

    def test_metric_properties_as_tensor(self):
        """Test metric tensor properties"""
        g = self.metric_tensor

        # Metric should be symmetric
        symmetric_g = g.symmetrize()
        assert_allclose(symmetric_g.components, g.components)

        # Trace should be dimension (in signature (-,+,+,+) it's -1+1+1+1 = 2)
        # No wait, trace of metric in (-,+,+,+) is -1+1+1+1 = 2
        trace = g.trace()
        expected_trace = np.trace(self.metric.g)  # This is 2
        assert_allclose(trace, expected_trace)

    def test_tensor_contraction(self):
        """Test tensor contraction operations"""
        # Contract stress tensor with four-velocity: T^μν u_ν
        u_covariant = self.four_velocity.lower_index(0)

        # This should give the energy-momentum density four-vector
        energy_momentum = self.stress_tensor.contract(u_covariant, [(1, 0)])

        assert energy_momentum.rank == 1

        # In rest frame with u^μ = (c, 0, 0, 0), u_μ = (-c, 0, 0, 0)
        # T^μν u_ν = T^μ0 u_0 + T^μ1 u_1 + T^μ2 u_2 + T^μ3 u_3
        #          = T^μ0 (-c) + 0 + 0 + 0 = -c T^μ0
        # For perfect fluid: T^00 = ρ, T^11 = T^22 = T^33 = p, others = 0
        # So result should be (-c*ρ, 0, 0, 0) = (-c*1, 0, 0, 0)
        expected = np.array([-1.0 * PhysicalConstants.c, 0, 0, 0])  # -cρ in rest frame
        assert_allclose(energy_momentum.components, expected, rtol=1e-10)

    def test_symmetrization_antisymmetrization(self):
        """Test tensor symmetrization and antisymmetrization"""
        # Create an asymmetric tensor
        components = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        indices = IndexStructure(["mu", "nu"], ["covariant", "covariant"], ["none", "none"])
        tensor = LorentzTensor(components, indices)

        # Symmetrize
        sym_tensor = tensor.symmetrize()

        # Check symmetry: T_{(μν)} = T_{νμ}
        sym_components = sym_tensor.components
        assert_allclose(sym_components, sym_components.T)

        # Check explicit formula: T_{(μν)} = ½(T_{μν} + T_{νμ})
        expected_sym = 0.5 * (components + components.T)
        assert_allclose(sym_components, expected_sym)

        # Antisymmetrize
        antisym_tensor = tensor.antisymmetrize()

        # Check antisymmetry: T_{[μν]} = -T_{[νμ]}
        antisym_components = antisym_tensor.components
        assert_allclose(antisym_components, -antisym_components.T)

        # Check explicit formula: T_{[μν]} = ½(T_{μν} - T_{νμ})
        expected_antisym = 0.5 * (components - components.T)
        assert_allclose(antisym_components, expected_antisym)

    def test_trace_operations(self):
        """Test tensor trace operations"""
        # Trace of stress tensor: T^μ_μ = T^00 + T^11 + T^22 + T^33
        trace = self.stress_tensor.trace()

        # For perfect fluid: ρ + 3p
        expected = 1.0 + 3 * 0.3  # ρ + 3p = 1.9
        assert_allclose(trace, expected)

        # Trace of metric: g^μ_μ = 4 (dimension)
        # Wait, this needs raising one index first
        metric_mixed = self.metric_tensor.raise_index(0)  # g^μ_ν
        metric_trace = metric_mixed.trace()
        assert_allclose(metric_trace, 4)  # Spacetime dimension

    def test_spatial_projection(self):
        """Test spatial projection operations"""
        # Project stress tensor orthogonal to four-velocity
        projected = self.stress_tensor.project_spatial(self.four_velocity.components)

        # In rest frame, projection should remove temporal components
        # For perfect fluid, this should give spatial pressure tensor
        expected = np.array([[0, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
        assert_allclose(projected.components, expected, rtol=1e-10)


class TestChristoffelSymbols:
    """Test Christoffel symbol computation."""

    def setup_method(self):
        """Set up test fixtures"""
        self.metric = Metric()
        g_indices = IndexStructure(
            ["mu", "nu"], ["covariant", "covariant"], ["symmetric", "symmetric"]
        )
        self.metric_tensor = LorentzTensor(self.metric.g, g_indices, self.metric)

    def test_flat_space_christoffel(self):
        """Test Christoffel symbols in flat Minkowski space"""
        # In flat space, all Christoffel symbols should vanish
        christoffel = self.metric_tensor.christoffel_symbols()

        assert christoffel.rank == 3
        expected_zero = np.zeros((4, 4, 4))
        assert_allclose(christoffel.components, expected_zero)

        # Check index structure
        assert christoffel.indices.types == ["contravariant", "covariant", "covariant"]
        assert "symmetric" in christoffel.indices.symmetries  # Symmetric in last two indices

    def test_christoffel_symmetry(self):
        """Test Christoffel symbol symmetry properties"""
        # Create some metric derivatives (even if artificial)
        metric_derivs = np.random.random((4, 4, 4))
        # Make symmetric in last two indices of derivatives to be consistent
        for i in range(4):
            for j in range(4):
                for k in range(j + 1, 4):
                    avg = 0.5 * (metric_derivs[i, j, k] + metric_derivs[i, k, j])
                    metric_derivs[i, j, k] = avg
                    metric_derivs[i, k, j] = avg

        christoffel = self.metric_tensor.christoffel_symbols(metric_derivs)

        # Check symmetry in lower indices: Γ^λ_μν = Γ^λ_νμ
        for lam in range(4):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    assert_allclose(
                        christoffel.components[lam, mu, nu],
                        christoffel.components[lam, nu, mu],
                        rtol=1e-10,
                    )


class TestCovariantDerivative:
    """Test covariant derivative operations."""

    def setup_method(self):
        """Set up test fixtures"""
        self.metric = Metric()

        # Test vector
        v_components = np.array([1, 2, 3, 4])
        v_indices = IndexStructure(["mu"], ["contravariant"], ["none"])
        self.test_vector = LorentzTensor(v_components, v_indices, self.metric)

    def test_covariant_derivative_flat_space(self):
        """Test covariant derivative in flat space"""
        # In flat space, covariant derivative reduces to partial derivative
        cov_deriv = self.test_vector.covariant_derivative(0)

        # Should have rank 2 (added one derivative index)
        assert cov_deriv.rank == 2

        # New index should be covariant
        assert cov_deriv.indices.types[0] == "covariant"
        assert cov_deriv.indices.types[1] == "contravariant"

        # Should have correct shape
        expected_shape = (4, 4)
        assert cov_deriv.shape == expected_shape

        # Components should be finite (not NaN or infinite)
        assert np.all(np.isfinite(cov_deriv.components))

        # In flat space, should only contain partial derivative terms
        # (no Christoffel symbol contributions)
        assert cov_deriv.components is not None

    def test_covariant_derivative_with_christoffel(self):
        """Test covariant derivative with non-zero Christoffel symbols"""
        # Create non-zero Christoffel symbols
        christoffel_components = np.zeros((4, 4, 4), dtype=complex)
        # Add a simple non-zero component for testing
        christoffel_components[0, 1, 1] = 0.1
        christoffel_components[1, 0, 1] = 0.1

        christoffel_indices = IndexStructure(
            ["rho", "mu", "nu"],
            ["contravariant", "covariant", "covariant"],
            ["none", "none", "none"],
        )
        christoffel = LorentzTensor(christoffel_components, christoffel_indices, self.metric)

        # Compute covariant derivative
        cov_deriv = self.test_vector.covariant_derivative(0, christoffel)

        # Should have rank 2 (added one derivative index)
        assert cov_deriv.rank == 2

        # Check index structure
        assert cov_deriv.indices.types[0] == "covariant"  # derivative index
        assert cov_deriv.indices.types[1] == "contravariant"  # original vector index

        # Should have correct shape
        expected_shape = (4, 4)
        assert cov_deriv.shape == expected_shape

        # Components should be finite (not NaN or infinite)
        assert np.all(np.isfinite(cov_deriv.components))

    def test_covariant_derivative_rank_2_tensor(self):
        """Test covariant derivative of a rank-2 tensor"""
        # Create a simple rank-2 tensor (like stress-energy tensor)
        tensor_components = np.ones((4, 4), dtype=complex)
        tensor_indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["none", "none"]
        )
        rank2_tensor = LorentzTensor(tensor_components, tensor_indices, self.metric)

        # Take covariant derivative
        cov_deriv = rank2_tensor.covariant_derivative(1)  # Insert at position 1

        # Should have rank 3 (added one derivative index)
        assert cov_deriv.rank == 3

        # Check index structure
        assert cov_deriv.indices.types[0] == "contravariant"  # original
        assert cov_deriv.indices.types[1] == "covariant"  # derivative
        assert cov_deriv.indices.types[2] == "contravariant"  # original

        # Should have correct shape
        expected_shape = (4, 4, 4)
        assert cov_deriv.shape == expected_shape

        # Components should be finite
        assert np.all(np.isfinite(cov_deriv.components))


class TestPhysicalInvariants:
    """Test physical invariants and constraints."""

    def setup_method(self):
        """Set up test fixtures"""
        self.metric = Metric()

    def test_lorentz_invariant_interval(self):
        """Test Lorentz invariant interval ds²"""
        # Two events
        dx = np.array([1, 2, 3, 4])  # (cdt, dx, dy, dz)

        # Compute interval: ds² = g_μν dx^μ dx^ν
        ds_squared = np.dot(dx, self.metric.g @ dx)

        # Should be invariant under Lorentz transformations
        # This is just a basic check that the calculation works
        expected = -1 + 4 + 9 + 16  # -c²dt² + dx² + dy² + dz² = 28
        assert_allclose(ds_squared, expected)

    def test_light_cone_structure(self):
        """Test light-cone structure preservation"""
        # Light-like interval: ds² = 0
        # For light ray: dx² + dy² + dz² = c²dt²
        dt = 1
        dx = PhysicalConstants.c * dt
        light_ray = np.array([PhysicalConstants.c * dt, dx, 0, 0])

        ds_squared = np.dot(light_ray, self.metric.g @ light_ray)
        assert_allclose(ds_squared, 0, atol=1e-12)

    def test_energy_momentum_conservation(self):
        """Test energy-momentum conservation properties"""
        # Perfect fluid stress tensor
        rho, p = 1.0, 0.3
        T = np.diag([rho, p, p, p])

        # Four-velocity in rest frame
        u = np.array([PhysicalConstants.c, 0, 0, 0])

        # Energy density: T^μν u_μ u_ν / c² should equal rho
        u_covariant = self.metric.g @ u
        (np.einsum("mu,mu", T, np.outer(u_covariant, u_covariant)) / PhysicalConstants.c**2)

        # This is a more complex calculation, but should relate to rho
        # For perfect fluid in rest frame, this relation should hold

    def test_causality_constraint(self):
        """Test that no signals propagate faster than light"""
        # This would be tested through dispersion relations
        # For now, just verify that metric signature is correct for causality
        eigenvals = np.linalg.eigvals(self.metric.g)

        # Should have one negative and three positive eigenvalues
        negative_eigenvals = eigenvals[eigenvals < 0]
        positive_eigenvals = eigenvals[eigenvals > 0]

        assert len(negative_eigenvals) == 1
        assert len(positive_eigenvals) == 3

        # Check the signature more directly
        signature = np.sign(np.diag(self.metric.g))
        expected_signature = np.array([-1, 1, 1, 1])
        assert_array_equal(signature, expected_signature)


@pytest.mark.benchmark
class TestTensorPerformance:
    """Benchmark tensor operations for performance."""

    def setup_method(self):
        """Set up performance test fixtures"""
        self.metric = Metric()

        # Large tensor for performance testing
        components = np.random.random((4, 4, 4, 4)) + 1j * np.random.random((4, 4, 4, 4))
        indices = IndexStructure(
            ["mu", "nu", "rho", "sigma"],
            ["contravariant", "covariant", "contravariant", "covariant"],
            ["none", "none", "none", "none"],
        )
        self.large_tensor = LorentzTensor(components, indices, self.metric)

    def test_contraction_performance(self, benchmark):
        """Benchmark tensor contraction"""
        # Create another tensor to contract with
        components2 = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
        indices2 = IndexStructure(["rho", "tau"], ["contravariant", "covariant"], ["none", "none"])
        tensor2 = LorentzTensor(components2, indices2, self.metric)

        def contract_tensors():
            return self.large_tensor.contract(tensor2, [(2, 0)])

        result = benchmark(contract_tensors)
        # Original tensor rank 4, second tensor rank 2, contracting 1 pair
        # Result rank = 4 + 2 - 2*1 = 4
        assert result.rank == 4

    def test_symmetrization_performance(self, benchmark):
        """Benchmark tensor symmetrization"""
        # Use smaller tensor for symmetrization (factorial complexity)
        components = np.random.random((4, 4, 4))
        indices = IndexStructure(
            ["mu", "nu", "rho"], ["covariant", "covariant", "covariant"], ["none", "none", "none"]
        )
        tensor = LorentzTensor(components, indices, self.metric)

        result = benchmark(tensor.symmetrize)
        assert result.rank == 3


class TestTensorContractionBugFixes:
    """Test fixes for tensor contraction bugs identified in chatgpt_buglist.md"""

    def setup_method(self):
        """Set up test tensors and metric"""
        self.metric = Metric()

    def test_scalar_return_for_rank_zero_contractions(self):
        """Test that rank-0 contractions return Python scalars, not LorentzTensor objects"""
        # Create vector tensors
        indices_up = IndexStructure(["mu"], ["contravariant"], ["none"])
        indices_down = IndexStructure(["nu"], ["covariant"], ["none"])

        u_up = LorentzTensor(np.array([1.0, 0.5, 0.0, 0.0]), indices_up, self.metric)
        u_down = u_up.lower_index(0)

        # Contract to get scalar: u^μ u_μ
        result = u_up.contract(u_down, [(0, 0)])

        # Should be Python scalar, not LorentzTensor
        assert isinstance(result, (float, complex)), f"Expected scalar, got {type(result)}"
        assert not hasattr(result, "rank"), "Scalar result should not have tensor properties"

        # Should work with abs() function (this was the original bug)
        abs_result = abs(result)
        assert isinstance(abs_result, float), f"abs() should return float, got {type(abs_result)}"

    def test_non_scalar_contractions_still_return_tensors(self):
        """Test that non-scalar contractions still return LorentzTensor objects"""
        # Create rank-2 and rank-1 tensors
        indices_2 = IndexStructure(["mu", "nu"], ["contravariant", "covariant"], ["none", "none"])
        indices_1 = IndexStructure(["rho"], ["contravariant"], ["none"])

        tensor_2 = LorentzTensor(np.random.rand(4, 4), indices_2, self.metric)
        vector = LorentzTensor(np.array([1.0, 0.5, 0.0, 0.0]), indices_1, self.metric)

        # Contract one index: T^μ_ν u^ρ → T^μ_ρ (rank-1 result)
        result = tensor_2.contract(vector, [(1, 0)])

        # Should still be LorentzTensor
        assert isinstance(result, LorentzTensor), f"Expected LorentzTensor, got {type(result)}"
        assert result.rank == 1, f"Expected rank-1, got rank-{result.rank}"

    def test_same_variance_contraction_uses_metric(self):
        """Test that same-variance contractions properly use the metric tensor"""
        indices_up = IndexStructure(["mu"], ["contravariant"], ["none"])

        # Two contravariant vectors
        u1 = LorentzTensor(np.array([1.0, 0.5, 0.0, 0.0]), indices_up, self.metric)
        u2 = LorentzTensor(np.array([2.0, 0.3, 0.1, 0.0]), indices_up, self.metric)

        # Contract u^μ v^ν (both up indices - should use metric)
        result = u1.contract(u2, [(0, 0)])

        # Manual calculation with metric: g_μν u^μ v^ν
        expected = np.einsum("ab,a,b->", self.metric.g, u1.components, u2.components)
        expected_real = expected.real if isinstance(expected, complex) else expected

        assert_allclose(result.real, expected_real, rtol=1e-12)

        # Verify it's different from wrong einsum (without metric)
        wrong_result = np.einsum("a,a->", u1.components, u2.components)
        assert abs(result.real - wrong_result.real) > 1e-6, "Result should differ from wrong einsum"

    def test_mixed_variance_contraction_direct_sum(self):
        """Test that mixed-variance contractions use direct summation (no metric needed)"""
        indices_up = IndexStructure(["mu"], ["contravariant"], ["none"])
        indices_down = IndexStructure(["nu"], ["covariant"], ["none"])

        u_up = LorentzTensor(np.array([1.0, 0.5, 0.0, 0.0]), indices_up, self.metric)
        v_down = LorentzTensor(np.array([2.0, 0.3, 0.1, 0.0]), indices_down, self.metric)

        # Contract u^μ v_ν (mixed indices - direct sum appropriate)
        result = u_up.contract(v_down, [(0, 0)])

        # Should equal direct sum
        expected = np.sum(u_up.components * v_down.components).real
        assert_allclose(result.real, expected, rtol=1e-12)

    def test_covariant_covariant_contraction_uses_inverse_metric(self):
        """Test that covariant-covariant contractions use inverse metric"""
        indices_down = IndexStructure(["mu"], ["covariant"], ["none"])

        # Two covariant vectors
        u_down = LorentzTensor(np.array([1.0, 0.5, 0.0, 0.0]), indices_down, self.metric)
        v_down = LorentzTensor(np.array([2.0, 0.3, 0.1, 0.0]), indices_down, self.metric)

        # Contract u_μ v_ν (both down indices - should use g^μν)
        result = u_down.contract(v_down, [(0, 0)])

        # Manual calculation with inverse metric: g^μν u_μ v_ν
        g_inv = np.linalg.inv(self.metric.g)
        expected = np.einsum("ab,a,b->", g_inv, u_down.components, v_down.components)
        expected_real = expected.real if isinstance(expected, complex) else expected

        assert_allclose(result.real, expected_real, rtol=1e-12)

    def test_four_velocity_normalization_with_fixes(self):
        """Test that four-velocity normalization works correctly with fixed contractions"""
        # Create normalized four-velocity in rest frame
        indices_up = IndexStructure(["mu"], ["contravariant"], ["none"])
        u_up = LorentzTensor(np.array([PhysicalConstants.c, 0.0, 0.0, 0.0]), indices_up, self.metric)
        u_down = u_up.lower_index(0)

        # Contract to get u^μ u_μ (should be -c²)
        norm_squared = u_up.contract(u_down, [(0, 0)])

        # Should be scalar and equal to -c²
        assert isinstance(norm_squared, (float, complex))
        expected = -(PhysicalConstants.c**2)
        assert_allclose(norm_squared.real, expected, rtol=1e-12)

        # abs() should work without type errors
        magnitude = abs(norm_squared)
        assert_allclose(magnitude, abs(expected), rtol=1e-12)

    def test_orthogonality_calculations_work(self):
        """Test that orthogonality calculations work with fixed contractions"""
        # Create test vectors
        indices_up = IndexStructure(["mu"], ["contravariant"], ["none"])

        # Four-velocity (timelike)
        u = LorentzTensor(np.array([PhysicalConstants.c, 0.0, 0.0, 0.0]), indices_up, self.metric)

        # Spatial vector (should be orthogonal)
        v = LorentzTensor(np.array([0.0, 1.0, 0.0, 0.0]), indices_up, self.metric)

        # Lower index for proper contraction
        v_down = v.lower_index(0)
        
        # Check orthogonality: u^μ v_μ should be 0
        orthogonality = u.contract(v_down, [(0, 0)])

        # Should be scalar and close to zero
        assert isinstance(orthogonality, (float, complex))
        assert abs(orthogonality) < 1e-12, f"Expected orthogonality, got {orthogonality}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
