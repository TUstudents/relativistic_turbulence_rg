"""
Test suite for hard-coded metric constraint fixes.

Tests that ConstrainedTensorField and PropagatorCalculator work correctly
with different metric dimensions and signatures.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rtrg.core.tensors import (
    ConstrainedTensorField,
    IndexType,
    Metric,
    TensorIndex,
    TensorIndexStructure,
)


class TestConstraintMetricFixes:
    """Test constraint helpers work with arbitrary metrics."""

    @pytest.fixture
    def minkowski_4d(self):
        """Standard 4D Minkowski metric."""
        return Metric()

    @pytest.fixture
    def minkowski_3d(self):
        """2+1D Minkowski metric."""
        return Metric(dimension=3, signature=(-1, 1, 1))

    @pytest.fixture
    def minkowski_5d(self):
        """4+1D Minkowski metric."""
        return Metric(dimension=5, signature=(-1, 1, 1, 1, 1))

    def test_4d_normalization(self, minkowski_4d):
        """Test normalization constraint with standard 4D metric."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField("u", structure, minkowski_4d, ["normalized"])

        # Test vector that's not normalized
        components = np.array([2.0, 0.5, 0.3, 0.1])
        normalized = field.apply_constraints(components)

        # Check normalization: u^μ u_μ = -1
        norm = np.einsum("i,ij,j->", normalized, minkowski_4d.g, normalized)
        assert abs(norm + 1.0) < 1e-10

    def test_3d_normalization(self, minkowski_3d):
        """Test normalization constraint with 2+1D metric."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField("u", structure, minkowski_3d, ["normalized"])

        # Test 3D vector
        components = np.array([2.0, 0.5, 0.3])
        normalized = field.apply_constraints(components)

        # Check normalization: u^μ u_μ = -1 in 3D
        norm = np.einsum("i,ij,j->", normalized, minkowski_3d.g, normalized)
        assert abs(norm + 1.0) < 1e-10
        assert len(normalized) == 3

    def test_5d_normalization(self, minkowski_5d):
        """Test normalization constraint with 4+1D metric."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField("u", structure, minkowski_5d, ["normalized"])

        # Test 5D vector
        components = np.array([2.0, 0.5, 0.3, 0.1, 0.2])
        normalized = field.apply_constraints(components)

        # Check normalization: u^μ u_μ = -1 in 5D
        norm = np.einsum("i,ij,j->", normalized, minkowski_5d.g, normalized)
        assert abs(norm + 1.0) < 1e-10
        assert len(normalized) == 5

    def test_velocity_orthogonality_4d(self, minkowski_4d):
        """Test velocity orthogonality in 4D."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField("u", structure, minkowski_4d, ["orthogonal_to_velocity"])

        # Test vector and velocity
        components = np.array([1.0, 2.0, 3.0, 4.0])
        velocity = np.array([1.0, 0.0, 0.0, 0.0])  # Rest frame

        orthogonal = field.apply_constraints(components, velocity=velocity)

        # Check orthogonality: q·u = 0
        dot_product = np.einsum("i,ij,j->", orthogonal, minkowski_4d.g, velocity)
        assert abs(dot_product) < 1e-12

    def test_velocity_orthogonality_3d(self, minkowski_3d):
        """Test velocity orthogonality in 3D."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField("u", structure, minkowski_3d, ["orthogonal_to_velocity"])

        # Test 3D vector and velocity
        components = np.array([1.0, 2.0, 3.0])
        velocity = np.array([1.0, 0.0, 0.0])  # 3D rest frame

        orthogonal = field.apply_constraints(components, velocity=velocity)

        # Check orthogonality: q·u = 0 in 3D
        dot_product = np.einsum("i,ij,j->", orthogonal, minkowski_3d.g, velocity)
        assert abs(dot_product) < 1e-12

    def test_validation_different_metrics(self, minkowski_3d, minkowski_4d, minkowski_5d):
        """Test constraint validation works with different metrics."""

        for metric, dim in [(minkowski_3d, 3), (minkowski_4d, 4), (minkowski_5d, 5)]:
            mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
            structure = TensorIndexStructure([mu_idx])
            field = ConstrainedTensorField("u", structure, metric, ["normalized"])

            # Create normalized vector
            normalized_components = np.zeros(dim)
            normalized_components[0] = 1.0  # Rest frame four-velocity

            validation = field.validate_constraints(normalized_components)
            assert validation["normalized"], f"Failed validation for {dim}D metric"

            # Create non-normalized vector
            non_normalized = normalized_components * 2.0
            validation = field.validate_constraints(non_normalized)
            assert not validation["normalized"], f"Should fail validation for {dim}D metric"

    def test_default_velocity_dimension_aware(self):
        """Test that default velocity respects metric dimension."""

        # Test different dimensions
        for dim in [3, 4, 5]:
            metric = Metric(dimension=dim)
            mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
            structure = TensorIndexStructure([mu_idx])
            field = ConstrainedTensorField("u", structure, metric, ["orthogonal_to_velocity"])

            # Get default velocity
            default_velocity = field._get_default_velocity()

            assert len(default_velocity) == dim
            assert default_velocity[0] == 1.0  # Time component
            assert all(default_velocity[1:] == 0.0)  # Spatial components zero

    def test_no_hardcoded_arrays(self, minkowski_3d):
        """Test that no hard-coded 4D arrays break 3D calculations."""
        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField(
            "u", structure, minkowski_3d, ["normalized", "orthogonal_to_velocity"]
        )

        # 3D vector
        components = np.array([1.5, 0.8, 0.6])
        velocity = np.array([1.0, 0.0, 0.0])

        # This would break if any hard-coded 4D arrays remained
        try:
            constrained = field.apply_constraints(components, velocity=velocity)
            validation = field.validate_constraints(constrained, velocity=velocity)

            # Should work without errors
            assert len(constrained) == 3
            assert "normalized" in validation
            assert "orthogonal_to_velocity" in validation

        except (ValueError, IndexError) as e:
            pytest.fail(f"Hard-coded arrays still present: {e}")


class TestMetricSignatureSupport:
    """Test support for different metric signatures."""

    def test_mostly_minus_signature(self):
        """Test mostly minus signature (+,-,-,-)."""
        # This would be useful for some theoretical studies
        metric = Metric(signature=(1, -1, -1, -1))

        mu_idx = TensorIndex("mu", IndexType.SPACETIME, "upper")
        structure = TensorIndexStructure([mu_idx])
        field = ConstrainedTensorField("u", structure, metric, ["normalized"])

        # In mostly minus, u^μ u_μ = +1 for timelike vectors
        components = np.array([2.0, 0.5, 0.3, 0.1])

        # Need to create field with custom norm_value for mostly minus
        field_mostly_minus = ConstrainedTensorField("u", structure, metric, ["normalized"])
        # Override the normalization method to use +1 instead of -1
        normalized = field_mostly_minus._apply_normalization(components, norm_value=1.0)

        # Check normalization for mostly minus signature
        norm = np.einsum("i,ij,j->", normalized, metric.g, normalized)
        assert abs(norm - 1.0) < 1e-10  # Should be +1, not -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
