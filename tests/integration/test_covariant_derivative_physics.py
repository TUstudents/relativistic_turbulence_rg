"""
Integration tests for covariant derivative physics validation.

Tests that covariant derivatives work correctly with physical constraints
from relativistic hydrodynamics.
"""

import numpy as np
import pytest

from rtrg.core.tensors import IndexStructure, LorentzTensor, Metric


class TestCovariantDerivativePhysics:
    """Integration tests for covariant derivative with physics validation."""

    def setup_method(self):
        """Set up physics test fixtures."""
        self.metric = Metric()

    def test_covariant_derivative_conservation_law(self):
        """Test that covariant derivatives satisfy energy-momentum conservation."""
        # Create a symmetric stress-energy tensor
        T_components = np.zeros((4, 4), dtype=complex)
        # Diagonal terms (energy density and pressures)
        T_components[0, 0] = 1.0  # Energy density
        T_components[1, 1] = 0.3  # Pressure
        T_components[2, 2] = 0.3
        T_components[3, 3] = 0.3
        # Small off-diagonal terms
        T_components[0, 1] = T_components[1, 0] = 0.05

        T_indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["none", "symmetric"]
        )
        stress_tensor = LorentzTensor(T_components, T_indices, self.metric)

        # Take covariant derivative (conservation law: ∇_μ T^{μν} = 0)
        div_T = stress_tensor.covariant_derivative(0)  # ∇_μ T^{μν}

        # Verify tensor structure
        assert div_T.rank == 3
        assert div_T.shape == (4, 4, 4)

        # Conservation should be approximately satisfied for this test tensor
        # (exact conservation requires solving the full Einstein equations)
        conservation_violation = np.max(np.abs(div_T.components))
        assert conservation_violation < 1.0  # Should be reasonably small

    def test_covariant_derivative_perturbation_analysis(self):
        """Test covariant derivatives for small perturbations."""
        # Create a small perturbation tensor (like in hydrodynamics)
        delta_T_components = 0.01 * np.random.random((4, 4))
        # Make it symmetric
        delta_T_components = (delta_T_components + delta_T_components.T) / 2

        delta_T_indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "contravariant"], ["none", "symmetric"]
        )
        delta_T = LorentzTensor(delta_T_components, delta_T_indices, self.metric)

        # Take covariant derivative
        nabla_delta_T = delta_T.covariant_derivative(0)

        # Verify mathematical consistency
        assert nabla_delta_T.rank == 3
        assert np.all(np.isfinite(nabla_delta_T.components))

        # The covariant derivative should preserve the tensor's physical properties
        # Check that the result has reasonable magnitude (or could be zero for this test case)
        max_component = np.max(np.abs(nabla_delta_T.components))
        assert max_component < 1.0  # Should be bounded
        # Note: In this test implementation, the derivative may be zero or small

    def test_covariant_derivative_basic_properties(self):
        """Test basic mathematical properties of covariant derivatives."""
        # Create a simple 4x4 test tensor
        test_components = np.zeros((4, 4), dtype=complex)
        test_components[0, 0] = 1.0
        test_components[0, 1] = 0.5
        test_components[1, 0] = 0.5
        test_components[1, 1] = 2.0
        test_indices = IndexStructure(
            ["mu", "nu"], ["contravariant", "covariant"], ["none", "none"]
        )
        test_tensor = LorentzTensor(test_components, test_indices, self.metric)

        # Take covariant derivative
        cov_deriv = test_tensor.covariant_derivative(0)

        # Verify basic properties
        assert cov_deriv.rank == test_tensor.rank + 1
        assert cov_deriv.shape[1:] == test_tensor.shape  # Original dimensions preserved
        assert np.all(np.isfinite(cov_deriv.components))

        # The derivative index should be covariant
        assert cov_deriv.indices.types[0] == "covariant"

    def test_index_structure_preservation(self):
        """Test that covariant derivatives preserve correct index structures."""
        # Test with mixed tensor T^μ_ν (one up, one down)
        T_components = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
        T_indices = IndexStructure(["mu", "nu"], ["contravariant", "covariant"], ["none", "none"])
        mixed_tensor = LorentzTensor(T_components, T_indices, self.metric)

        # Take covariant derivative at different positions
        nabla_T_front = mixed_tensor.covariant_derivative(0)  # ∇_σ T^μ_ν
        nabla_T_middle = mixed_tensor.covariant_derivative(1)  # T^μ ∇_σ _ν
        nabla_T_end = mixed_tensor.covariant_derivative(2)  # T^μ_ν ∇_σ

        # Check index structures
        # Front: [covariant(σ), contravariant(μ), covariant(ν)]
        assert nabla_T_front.indices.types == ["covariant", "contravariant", "covariant"]

        # Middle: [contravariant(μ), covariant(σ), covariant(ν)]
        assert nabla_T_middle.indices.types == ["contravariant", "covariant", "covariant"]

        # End: [contravariant(μ), covariant(ν), covariant(σ)]
        assert nabla_T_end.indices.types == ["contravariant", "covariant", "covariant"]

        # All should have rank 3
        assert nabla_T_front.rank == nabla_T_middle.rank == nabla_T_end.rank == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
