"""
Unit tests for Task 3: Critical Bug Fixes

This module tests the fixes for the critical bugs identified in Task 3:
1. Spatial projector idempotency issue
2. Metric dimension handling bug
3. Comprehensive constraint validation system

All fixes should maintain backward compatibility while resolving the identified issues.
"""

import numpy as np
import pytest

from rtrg.core.tensors import Metric
from rtrg.israel_stewart.constraints import (
    ValidationReport,
    VelocityConstraint,
    apply_constraint_corrections,
    constraint_violation_magnitude,
    spatial_projector,
    validate_all_constraints,
    verify_projector_idempotency,
)


class TestSpatialProjectorIdempotency:
    """Test spatial projector idempotency fix."""

    def test_rest_frame_idempotency(self):
        """Test spatial projector idempotency in rest frame."""
        metric = Metric()
        u_rest = np.array([1.0, 0.0, 0.0, 0.0])

        h_proj = spatial_projector(u_rest, metric)
        h_squared = h_proj @ h_proj

        max_error = np.max(np.abs(h_squared - h_proj))
        assert max_error < 1e-14, f"Rest frame idempotency failed: max error = {max_error}"

    def test_moving_frame_idempotency(self):
        """Test spatial projector idempotency in moving frame."""
        metric = Metric()

        # Properly normalized moving frame
        v_spatial = np.array([0.1, 0.0, 0.0])
        gamma = 1.0 / np.sqrt(1 - np.sum(v_spatial**2))
        u_moving = np.array([gamma, gamma * v_spatial[0], 0.0, 0.0])

        h_proj = spatial_projector(u_moving, metric)
        h_squared = h_proj @ h_proj

        max_error = np.max(np.abs(h_squared - h_proj))
        assert max_error < 1e-12, f"Moving frame idempotency failed: max error = {max_error}"

    def test_boosted_frame_idempotency(self):
        """Test spatial projector idempotency in general boosted frame."""
        metric = Metric()

        # Properly normalized boosted frame
        v_spatial = np.array([0.3, 0.2, 0.1])
        v_sq = np.sum(v_spatial**2)
        gamma = 1.0 / np.sqrt(1 - v_sq)
        u_boosted = np.array(
            [gamma, gamma * v_spatial[0], gamma * v_spatial[1], gamma * v_spatial[2]]
        )

        h_proj = spatial_projector(u_boosted, metric)
        h_squared = h_proj @ h_proj

        max_error = np.max(np.abs(h_squared - h_proj))
        assert max_error < 1e-12, f"Boosted frame idempotency failed: max error = {max_error}"

    def test_high_velocity_idempotency(self):
        """Test spatial projector idempotency for high velocities (but < c)."""
        metric = Metric()

        # High velocity but still less than c
        v_spatial = np.array([0.9, 0.0, 0.0])
        gamma = 1.0 / np.sqrt(1 - np.sum(v_spatial**2))
        u_high_v = np.array([gamma, gamma * v_spatial[0], 0.0, 0.0])

        h_proj = spatial_projector(u_high_v, metric)
        h_squared = h_proj @ h_proj

        max_error = np.max(np.abs(h_squared - h_proj))
        assert max_error < 1e-12, f"High velocity idempotency failed: max error = {max_error}"

    def test_orthogonality_condition(self):
        """Test that spatial projector satisfies orthogonality to four-velocity."""
        metric = Metric()

        # Test various frames
        test_velocities = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # Rest
            np.array([1.005, 0.1, 0.0, 0.0]),  # Moving (as in bug report)
            np.array([1.15, 0.5, 0.3, 0.1]),  # Boosted
        ]

        for u in test_velocities:
            # Renormalize to ensure proper four-velocity
            if not np.allclose(u[0], 1.0) or np.any(u[1:] != 0):
                v_spatial = u[1:4]  # Extract spatial part
                if np.sum(v_spatial**2) < 1.0:  # Ensure v < c
                    gamma = 1.0 / np.sqrt(1 - np.sum(v_spatial**2))
                    u = np.array(
                        [gamma, gamma * v_spatial[0], gamma * v_spatial[1], gamma * v_spatial[2]]
                    )

            h_proj = spatial_projector(u, metric)
            u_lower = metric.g @ u

            # Check h^μ_ν u^ν = 0 for all μ (correct orthogonality condition)
            hu = h_proj @ u  # h^μ_ν u^ν
            max_violation = np.max(np.abs(hu))

            assert (
                max_violation < 1e-12
            ), f"Orthogonality condition violated: max |h·u| = {max_violation}"


class TestMetricDimensionHandling:
    """Test metric dimension handling fix."""

    def test_default_4d_metric(self):
        """Test that default 4D metric still works correctly."""
        metric = Metric()

        assert metric.dim == 4
        assert metric.signature == (-1, 1, 1, 1)
        assert metric.g.shape == (4, 4)

        expected_diag = np.array([-1, 1, 1, 1])
        assert np.allclose(np.diag(metric.g), expected_diag)

    @pytest.mark.parametrize("dimension", [2, 3, 5, 6, 8])
    def test_arbitrary_dimension_metrics(self, dimension):
        """Test metric construction for arbitrary dimensions."""
        metric = Metric(dimension=dimension)

        assert metric.dim == dimension
        assert len(metric.signature) == dimension
        assert metric.g.shape == (dimension, dimension)

        # Check signature is (-1, 1, 1, ..., 1)
        expected_signature = (-1,) + (1,) * (dimension - 1)
        assert metric.signature == expected_signature

        # Check diagonal elements
        expected_diag = np.array(expected_signature)
        assert np.allclose(np.diag(metric.g), expected_diag)

        # Check off-diagonal elements are zero
        off_diag = metric.g - np.diag(np.diag(metric.g))
        assert np.allclose(off_diag, 0.0)

    def test_custom_signature(self):
        """Test that custom signatures still work correctly."""
        custom_signature = (-1, -1, 1, 1)  # Different signature
        metric = Metric(dimension=4, signature=custom_signature)

        assert metric.signature == custom_signature
        assert np.allclose(np.diag(metric.g), np.array(custom_signature))

    def test_signature_dimension_mismatch_error(self):
        """Test that signature-dimension mismatches are properly detected."""
        with pytest.raises(ValueError, match="Signature length .* doesn't match dimension"):
            Metric(dimension=5, signature=(-1, 1, 1, 1))  # 4-element signature for 5D

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            Metric(dimension=0)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            Metric(dimension=-1)

    def test_higher_dimensional_spatial_projector(self):
        """Test that spatial projector works with higher-dimensional metrics."""
        for dim in [3, 5, 6]:
            metric = Metric(dimension=dim)

            # Create properly normalized four-velocity for this dimension
            u = np.zeros(dim)
            u[0] = 1.0  # Rest frame
            if dim > 1:
                u[1] = 0.1  # Small spatial component
                # Renormalize
                spatial_part = u[1:]
                v_sq = np.sum(spatial_part**2)
                if v_sq < 1.0:
                    gamma = 1.0 / np.sqrt(1 - v_sq)
                    u[0] = gamma
                    u[1:] = gamma * spatial_part

            # Test spatial projector construction and idempotency
            h_proj = spatial_projector(u, metric)
            assert h_proj.shape == (dim, dim)

            # Test idempotency
            h_squared = h_proj @ h_proj
            max_error = np.max(np.abs(h_squared - h_proj))
            assert (
                max_error < 1e-12
            ), f"{dim}D spatial projector not idempotent: error = {max_error}"


class TestConstraintValidationSystem:
    """Test comprehensive constraint validation system."""

    def create_valid_state(self, u: np.ndarray, metric: Metric) -> dict:
        """Create a state that satisfies all constraints."""
        dim = metric.dim

        # Create shear tensor that is symmetric, traceless, and spatial
        pi = np.zeros((dim, dim))
        if dim >= 3:
            # Only set spatial-spatial components
            pi[1, 2] = pi[2, 1] = 0.01  # Symmetric
            if dim == 4:
                pi[1, 1] = 0.02
                pi[2, 2] = -0.02  # Traceless in spatial subspace

        # Create spatial heat flux (orthogonal to u)
        q = np.zeros(dim)
        if dim >= 2:
            q[1] = 0.01  # Spatial component only

        return {
            "pi": pi,
            "q": q,
            "Pi": 0.05,  # Scalar bulk pressure
        }

    def test_valid_state_validation(self):
        """Test that properly constructed states pass all constraints."""
        metric = Metric()
        u_rest = np.array([1.0, 0.0, 0.0, 0.0])

        state = self.create_valid_state(u_rest, metric)
        report = validate_all_constraints(state, u_rest, metric)

        assert report.all_satisfied, f"Valid state failed validation: {report.summary()}"
        assert all(report.individual_results.values()), "Some individual constraints failed"

    def test_velocity_normalization_detection(self):
        """Test detection of four-velocity normalization violations."""
        metric = Metric()
        u_bad = np.array([1.1, 0.0, 0.0, 0.0])  # Not normalized

        state = self.create_valid_state(u_bad, metric)
        report = validate_all_constraints(state, u_bad, metric)

        assert not report.individual_results["velocity_normalization"]
        assert report.violation_magnitudes["velocity_normalization"] > 1e-10

    def test_spatial_projector_idempotency_detection(self):
        """Test that spatial projector idempotency violations are detected (this should not happen with our fix)."""
        metric = Metric()
        u_moving = np.array([1.005, 0.1, 0.0, 0.0])

        # Properly normalize
        v_spatial = u_moving[1:4]
        v_sq = np.sum(v_spatial**2)
        if v_sq < 1.0:
            gamma = 1.0 / np.sqrt(1 - v_sq)
            u_moving = np.array(
                [gamma, gamma * v_spatial[0], gamma * v_spatial[1], gamma * v_spatial[2]]
            )

        state = self.create_valid_state(u_moving, metric)
        report = validate_all_constraints(state, u_moving, metric)

        # With our fix, spatial projector should be idempotent
        assert report.individual_results[
            "spatial_projector_idempotency"
        ], f"Spatial projector idempotency failed after fix: violation = {report.violation_magnitudes['spatial_projector_idempotency']}"

    def test_constraint_violation_quantification(self):
        """Test constraint violation magnitude quantification."""
        metric = Metric()
        u = np.array([1.0, 0.0, 0.0, 0.0])

        # Create state with known violations
        state = {
            "pi": np.array(
                [
                    [0.1, 0.0, 0.0, 0.0],  # Non-zero temporal component (violates spatial)
                    [0.0, 0.0, 0.05, 0.0],  # Asymmetric
                    [0.0, 0.1, 0.0, 0.0],  # Asymmetric
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "q": np.array([0.1, 0.0, 0.0, 0.0]),  # Non-zero temporal component
            "Pi": 0.0,
        }

        violations = constraint_violation_magnitude(state, u, metric)

        # Check that violations are properly quantified
        assert violations["shear_tensor_symmetric"] > 0.01  # Should detect asymmetry
        assert violations["heat_flux_spatial"] > 0.01  # Should detect temporal component

    def test_higher_dimensional_validation(self):
        """Test constraint validation in higher dimensions."""
        for dim in [3, 5]:
            metric = Metric(dimension=dim)
            u = np.zeros(dim)
            u[0] = 1.0  # Rest frame

            state = self.create_valid_state(u, metric)
            report = validate_all_constraints(state, u, metric)

            # Should pass basic constraints
            assert report.individual_results["velocity_normalization"]
            assert report.individual_results["spatial_projector_idempotency"]

    def test_constraint_correction_system(self):
        """Test the automatic constraint correction system."""
        metric = Metric()
        u = np.array([1.0, 0.0, 0.0, 0.0])

        # Create a state with mild constraint violations
        bad_state = {
            "pi": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.02, 0.01, 0.0],
                    [0.0, 0.015, -0.01, 0.0],  # Slightly asymmetric
                    [0.0, 0.0, 0.0, -0.01],
                ]
            ),  # Small trace
            "q": np.array([0.001, 0.01, 0.0, 0.0]),  # Small temporal component
            "Pi": 0.0,
        }

        corrected_state, report = apply_constraint_corrections(bad_state, u, metric)

        # Should improve constraint satisfaction
        original_report = validate_all_constraints(bad_state, u, metric)
        original_passed = sum(original_report.individual_results.values())
        corrected_passed = sum(report.individual_results.values())

        assert (
            corrected_passed >= original_passed
        ), "Constraint correction did not improve satisfaction"

    def test_validation_report_functionality(self):
        """Test ValidationReport class functionality."""
        # Create a simple report
        report = ValidationReport(
            all_satisfied=False,
            individual_results={"test1": True, "test2": False},
            violation_magnitudes={"test1": 0.0, "test2": 0.1},
            tolerance=1e-12,
            total_constraints=2,
        )

        summary = report.summary()
        assert "1/2 constraints" in summary
        assert "❌ FAIL" in summary
        assert "test2: violation = 1.00e-01" in summary


class TestBackwardCompatibility:
    """Test that fixes maintain backward compatibility."""

    def test_spatial_projector_api_compatibility(self):
        """Test that spatial_projector API is backward compatible."""
        metric = Metric()
        u = np.array([1.0, 0.0, 0.0, 0.0])

        # Should work with same API as before
        h_proj = spatial_projector(u, metric)

        # Should return 4x4 matrix
        assert h_proj.shape == (4, 4)
        assert isinstance(h_proj, np.ndarray)

    def test_metric_api_compatibility(self):
        """Test that Metric class API is backward compatible."""
        # Default construction should work as before
        metric = Metric()
        assert metric.dim == 4
        assert hasattr(metric, "g")
        assert hasattr(metric, "signature")

        # With custom signature should work as before
        metric_custom = Metric(dimension=4, signature=(-1, 1, 1, 1))
        assert metric_custom.dim == 4

    def test_velocity_constraint_compatibility(self):
        """Test that VelocityConstraint class is backward compatible."""
        vc = VelocityConstraint()

        # Should have all expected methods
        assert hasattr(vc, "is_satisfied")
        assert hasattr(vc, "normalize")
        assert hasattr(vc, "projector")
        assert hasattr(vc, "tt_projector")

        # Methods should work as before
        u = np.array([1.0, 0.0, 0.0, 0.0])
        assert vc.is_satisfied(u)

        h_proj = vc.projector(u)
        assert h_proj.shape == (4, 4)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_vector_handling(self):
        """Test handling of zero and near-zero vectors."""
        metric = Metric()

        # Zero four-velocity should issue a warning but still compute
        u_zero = np.zeros(4)

        with pytest.warns(UserWarning, match="Four-velocity not properly normalized"):
            h_proj = spatial_projector(u_zero, metric)
            # Should return a projector (though physically meaningless)
            assert h_proj.shape == (4, 4)
            assert isinstance(h_proj, np.ndarray)

    def test_superluminal_velocity_handling(self):
        """Test handling of superluminal velocities."""
        vc = VelocityConstraint()

        # Spatial velocity > c should raise error
        v_super = np.array([1.1, 0.0, 0.0])  # |v| > c

        with pytest.raises(ValueError, match="exceeds speed of light"):
            vc.normalize(v_super)

    def test_invalid_tensor_shapes(self):
        """Test handling of invalid tensor shapes."""
        metric = Metric()
        u = np.array([1.0, 0.0, 0.0, 0.0])

        # Wrong shape shear tensor
        bad_state = {
            "pi": np.array([[1, 2], [3, 4]]),  # Wrong shape
            "q": u,
            "Pi": 0.0,
        }

        # Should handle gracefully (skip malformed tensors)
        report = validate_all_constraints(bad_state, u, metric)
        assert isinstance(report, ValidationReport)

    def test_dimension_consistency(self):
        """Test that all operations are consistent across dimensions."""
        for dim in [2, 3, 4, 5]:
            metric = Metric(dimension=dim)
            u = np.zeros(dim)
            u[0] = 1.0

            # All operations should work consistently
            h_proj = spatial_projector(u, metric)
            assert h_proj.shape == (dim, dim)

            # Validation should work
            state = {"Pi": 0.0}  # Minimal valid state
            report = validate_all_constraints(state, u, metric)
            assert isinstance(report, ValidationReport)
