"""
Constraint handling utilities for Israel–Stewart (IS) hydrodynamics.

This module provides:
    - Four-velocity normalization checks and helpers
    - Spatial projector Δ^{μν} orthogonal to u^μ
    - Transverse–traceless (TT) projector P_{μναβ} for shear tensors
    - Convenience checks for spatial vectors and symmetric traceless tensors

Conventions:
    - Metric signature: (-, +, +, +)
    - Natural units with c = 1 unless specified via PhysicalConstants
    - Arrays are assumed to be in a coordinate basis with components indexed 0..3

Note:
    These helpers are light-weight and NumPy-based to be usable both in symbolic
    and numerical contexts. For advanced tensor operations, prefer rtrg.core.tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.constants import PhysicalConstants
from ..core.tensors import Metric


def spatial_projector(u: np.ndarray, metric: Metric | None = None) -> np.ndarray:
    """Return spatial projector Δ_{μν} orthogonal to u^μ.

    Δ_{μν} = g_{μν} + u_μ u_ν / c^2, where u_μ = g_{μσ} u^σ

    Args:
        u: Four-velocity components (contravariant). Must be length equal to metric.dim.
        metric: Spacetime metric. Defaults to Minkowski.

    Returns:
        2D array Δ_{μν} (covariant indices).
    """
    m = metric or Metric()
    u = np.asarray(u, dtype=float)
    if u.shape != (m.dim,):
        raise ValueError(f"four-velocity must have shape ({m.dim},), got {u.shape}")

    # Lower index: u_μ = g_{μν} u^ν
    u_lower = m.g @ u
    c2 = PhysicalConstants.c**2
    Delta = m.g + np.outer(u_lower, u_lower) / c2
    return Delta


def tt_projector(u: np.ndarray, metric: Metric | None = None) -> np.ndarray:
    """Return transverse–traceless projector P_{μναβ} for rank-2 shear tensors.

    P_{μναβ} = 1/2 (Δ_{μα} Δ_{νβ} + Δ_{μβ} Δ_{να}) - 1/(d-1) Δ_{μν} Δ_{αβ}

    where d is the spacetime dimension and d-1 is the number of spatial dimensions.

    Args:
        u: Four-velocity (contravariant)
        metric: Spacetime metric (defaults to Minkowski)

    Returns:
        4D array with indices (μ, ν, α, β) in covariant position.
    """
    m = metric or Metric()
    Delta = spatial_projector(u, m)
    dim = m.dim
    P = np.zeros((dim, dim, dim, dim), dtype=float)

    # Build P via explicit index operations
    # P_{μναβ} = 1/2 (Δ_{μα} Δ_{νβ} + Δ_{μβ} Δ_{να}) - 1/(d-1) Δ_{μν} Δ_{αβ}
    # where d-1 is the number of spatial dimensions
    spatial_dims = dim - 1  # Exclude time dimension
    term1 = 0.5 * (np.einsum("ma,nb->mnab", Delta, Delta) + np.einsum("mb,na->mnab", Delta, Delta))
    term2 = (1.0 / spatial_dims) * np.einsum("mn,ab->mnab", Delta, Delta)
    P = term1 - term2
    return np.asarray(P, dtype=float)


def is_symmetric_traceless_spatial(
    pi: np.ndarray, u: np.ndarray, metric: Metric | None = None, atol: float = 1e-12
) -> tuple[bool, dict]:
    """Check whether a rank-2 tensor satisfies shear constraints.

    Checks:
        - Symmetric: π_{μν} = π_{νμ}
        - Traceless: g^{μν} π_{μν} = 0 (treat input as covariant)
        - Spatial: u^μ π_{μν} = 0 = π_{μν} u^ν

    Args:
        pi: 4x4 array representing covariant components π_{μν}
        u: Four-velocity (contravariant)
        metric: Metric (defaults Minkowski)
        atol: Numerical tolerance for checks

    Returns:
        (ok, details) where ok indicates all constraints satisfied and details
        contains individual booleans.
    """
    m = metric or Metric()
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (m.dim, m.dim):
        raise ValueError(f"shear tensor must have shape {(m.dim, m.dim)}, got {pi.shape}")

    # Symmetry
    symmetric = np.allclose(pi, pi.T, atol=atol)

    # Traceless: use g^{μν} π_{μν}
    g_inv = np.linalg.inv(m.g)
    trace = float(np.einsum("mn,mn->", g_inv, pi))
    traceless = abs(trace) <= atol

    # Spatial: u^μ π_{μν} = 0 and π_{μν} u^ν = 0
    u_pi = np.einsum("m,mn->n", u, pi)  # u^μ π_{μν}
    pi_u = np.einsum("mn,n->m", pi, u)  # π_{μν} u^ν
    spatial_left = np.allclose(u_pi, 0.0, atol=atol)
    spatial_right = np.allclose(pi_u, 0.0, atol=atol)
    spatial = spatial_left and spatial_right

    ok = symmetric and traceless and spatial
    return ok, {
        "symmetric": symmetric,
        "traceless": traceless,
        "spatial_left": spatial_left,
        "spatial_right": spatial_right,
    }


def is_spatial_vector(
    q: np.ndarray, u: np.ndarray, metric: Metric | None = None, atol: float = 1e-12
) -> bool:
    """Check orthogonality u_μ q^μ = 0 for a vector q^μ.

    Args:
        q: Vector components q^μ (contravariant)
        u: Four-velocity (contravariant)
        metric: Metric (defaults Minkowski)
        atol: Numerical tolerance
    """
    m = metric or Metric()
    q = np.asarray(q, dtype=float)
    if q.shape != (m.dim,):
        raise ValueError(f"vector must have shape ({m.dim},), got {q.shape}")
    dot = float(np.dot(m.g @ u, q))
    return abs(dot) <= atol


@dataclass
class VelocityConstraint:
    """Four-velocity normalization and helper operations.

    Enforces u^μ u_μ = -c^2 and provides normalization from spatial velocity.
    """

    c: float = PhysicalConstants.c
    metric: Metric = Metric()

    def is_satisfied(self, u: np.ndarray, atol: float = 1e-12) -> bool:
        """Check normalization u·u = -c^2.

        Args:
            u: Four-velocity (contravariant)
            atol: Tolerance
        """
        u = np.asarray(u, dtype=float)
        if u.shape != (self.metric.dim,):
            return False
        u_dot = float(np.dot(u, self.metric.g @ u))
        return abs(u_dot + self.c**2) <= atol

    def normalize(self, v_spatial: np.ndarray) -> np.ndarray:
        """Build normalized four-velocity u^μ = γ(c, v^i).

        Args:
            v_spatial: Three-velocity (v_x, v_y, v_z) in same units as c.

        Returns:
            u: Four-velocity (γ c, γ v_x, γ v_y, γ v_z)
        """
        v_spatial = np.asarray(v_spatial, dtype=float)
        if v_spatial.shape != (3,):
            raise ValueError("spatial velocity must be 3-dimensional")

        v2 = float(np.dot(v_spatial, v_spatial))
        if v2 >= self.c**2:
            raise ValueError("spatial velocity magnitude exceeds speed of light")

        gamma = 1.0 / np.sqrt(1.0 - v2 / self.c**2)
        u0 = gamma * self.c
        ui = gamma * v_spatial
        u = np.array([u0, *ui.tolist()], dtype=float)
        return u

    def projector(self, u: np.ndarray) -> np.ndarray:
        """Convenience: spatial projector Δ_{μν} for a given u^μ."""
        return spatial_projector(u, self.metric)

    def tt_projector(self, u: np.ndarray) -> np.ndarray:
        """Convenience: TT projector P_{μναβ} for a given u^μ."""
        return tt_projector(u, self.metric)
