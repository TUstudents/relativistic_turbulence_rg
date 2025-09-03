"""
Unit tests for constraint utilities and projectors.
"""

import numpy as np
import pytest

from rtrg.core.constants import PhysicalConstants
from rtrg.core.tensors import Metric
from rtrg.israel_stewart.constraints import (
    VelocityConstraint,
    is_spatial_vector,
    is_symmetric_traceless_spatial,
    spatial_projector,
    tt_projector,
)


@pytest.mark.unit
class TestVelocityConstraint:
    def test_rest_frame_normalization(self):
        vc = VelocityConstraint()
        u = np.array([PhysicalConstants.c, 0.0, 0.0, 0.0])
        assert vc.is_satisfied(u)

    def test_normalize_from_spatial_velocity(self):
        vc = VelocityConstraint()
        v = np.array([0.3, 0.0, 0.0])
        u = vc.normalize(v)
        assert vc.is_satisfied(u)
        # gamma check
        c = PhysicalConstants.c
        gamma = 1.0 / np.sqrt(1.0 - (np.dot(v, v) / c**2))
        assert abs(u[0] - gamma * c) < 1e-12
        assert abs(u[1] - gamma * v[0]) < 1e-12

    def test_superluminal_rejected(self):
        vc = VelocityConstraint()
        with pytest.raises(ValueError):
            vc.normalize(np.array([1.5, 0.0, 0.0]))


@pytest.mark.unit
class TestProjectors:
    def test_spatial_projector_rest_frame(self):
        metric = Metric()
        u = np.array([PhysicalConstants.c, 0.0, 0.0, 0.0])
        Delta = spatial_projector(u, metric)
        # Δ should be diag(0, 1, 1, 1) in rest frame
        expected = np.diag([0.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(Delta, expected, atol=1e-12)

    def test_tt_projector_symmetry(self):
        metric = Metric()
        u = np.array([PhysicalConstants.c, 0.0, 0.0, 0.0])
        P = tt_projector(u, metric)
        # Symmetries: P_{μναβ} = P_{νμαβ} = P_{μνβα}
        np.testing.assert_allclose(P.transpose(1, 0, 2, 3), P, atol=1e-12)
        np.testing.assert_allclose(P.transpose(0, 1, 3, 2), P, atol=1e-12)

    def test_tt_projector_traceless(self):
        # Contract with g^{μν} should vanish: g^{μν} P_{μναβ} = 0
        metric = Metric()
        g_inv = np.linalg.inv(metric.g)
        u = np.array([PhysicalConstants.c, 0.0, 0.0, 0.0])
        P = tt_projector(u, metric)
        contracted = np.einsum("mn,mnab->ab", g_inv, P)
        np.testing.assert_allclose(contracted, np.zeros_like(contracted), atol=1e-12)


@pytest.mark.unit
class TestTensorAndVectorConstraints:
    def test_is_spatial_vector(self):
        metric = Metric()
        u = np.array([PhysicalConstants.c, 0.0, 0.0, 0.0])
        q_spatial = np.array([0.0, 1.0, 0.0, 0.0])
        assert is_spatial_vector(q_spatial, u, metric)

    def test_is_symmetric_traceless_spatial(self):
        metric = Metric()
        u = np.array([PhysicalConstants.c, 0.0, 0.0, 0.0])
        # Build a symmetric traceless spatial tensor (only spatial indices non-zero)
        pi = np.zeros((4, 4))
        pi[1, 2] = pi[2, 1] = 0.1
        pi[1, 1] = 0.05
        pi[2, 2] = -0.05  # traceless in spatial block
        ok, details = is_symmetric_traceless_spatial(pi, u, metric)
        assert ok
        assert details["symmetric"]
        assert details["traceless"]
        assert details["spatial_left"] and details["spatial_right"]
