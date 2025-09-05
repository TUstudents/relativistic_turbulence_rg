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
from typing import Any, Callable

import numpy as np

from ..core.constants import PhysicalConstants
from ..core.tensors import Metric


def spatial_projector(u: np.ndarray, metric: Metric | None = None) -> np.ndarray:
    """Return spatial projector h^μ_ν orthogonal to u^μ.

    The spatial projector projects tensors onto the spatial hypersurface orthogonal
    to the four-velocity. The correct formula with mixed indices is:
    h^μ_ν = δ^μ_ν + u^μ u_ν / c²

    This satisfies the idempotency condition h^μ_ρ h^ρ_ν = h^μ_ν and the
    orthogonality condition h^μ_ν u^ν = 0.

    Args:
        u: Four-velocity components (contravariant). Must be length equal to metric.dim.
            Should satisfy normalization u^μ u_μ = -c².
        metric: Spacetime metric. Defaults to Minkowski.

    Returns:
        2D array h^μ_ν (mixed indices: contravariant row, covariant column).
    """
    m = metric or Metric()
    u = np.asarray(u, dtype=float)
    if u.shape != (m.dim,):
        raise ValueError(f"four-velocity must have shape ({m.dim},), got {u.shape}")

    # Verify four-velocity normalization
    u_lower = m.g @ u
    u_norm_squared = np.dot(u, u_lower)
    c2 = PhysicalConstants.c**2

    expected_norm = -c2
    if abs(u_norm_squared - expected_norm) > 1e-10:
        import warnings

        warnings.warn(
            f"Four-velocity not properly normalized: u·u = {u_norm_squared:.6f}, "
            f"expected {expected_norm:.6f}. This may cause idempotency violations.",
            stacklevel=2,
        )

    # Correct spatial projector: h^μ_ν = δ^μ_ν + u^μ u_ν / c²
    # Since u·u = -c², this becomes: h^μ_ν = δ^μ_ν - u^μ u_ν / (u·u)
    # Which simplifies to: h^μ_ν = δ^μ_ν + u^μ u_ν for our normalization
    h_projector = np.eye(m.dim) + np.outer(u, u_lower) / c2
    return h_projector


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


@dataclass
class ValidationReport:
    """Report containing constraint validation results.

    Provides detailed information about which constraints are satisfied
    and quantifies violation magnitudes for debugging purposes.
    """

    all_satisfied: bool
    individual_results: dict[str, bool]
    violation_magnitudes: dict[str, float]
    tolerance: float
    total_constraints: int

    def summary(self) -> str:
        """Generate human-readable summary of validation results."""
        passed = sum(self.individual_results.values())
        total = self.total_constraints

        summary = "Constraint Validation Summary:\n"
        summary += f"- Passed: {passed}/{total} constraints\n"
        summary += f"- Overall: {'✅ PASS' if self.all_satisfied else '❌ FAIL'}\n"
        summary += f"- Tolerance: {self.tolerance:.2e}\n"

        if not self.all_satisfied:
            summary += "\nFailed Constraints:\n"
            for name, satisfied in self.individual_results.items():
                if not satisfied:
                    violation = self.violation_magnitudes.get(name, 0.0)
                    summary += f"  - {name}: violation = {violation:.2e}\n"

        return summary


def validate_all_constraints(
    state: dict[str, np.ndarray],
    u: np.ndarray,
    metric: Metric | None = None,
    tolerance: float = 1e-12,
) -> ValidationReport:
    """Validate all Israel-Stewart constraints simultaneously.

    Checks multiple constraint types for a given hydrodynamic state:
    - Four-velocity normalization: u^μ u_μ = -c²
    - Spatial projector idempotency: h² = h
    - Shear tensor constraints: π symmetric, traceless, spatial
    - Heat flux orthogonality: q^μ u_μ = 0
    - Bulk pressure constraints: Π scalar

    Args:
        state: Dictionary containing field values (pi, q, Pi, etc.)
        u: Four-velocity array
        metric: Metric tensor (defaults to Minkowski)
        tolerance: Numerical tolerance for constraint satisfaction

    Returns:
        ValidationReport with detailed constraint validation results
    """
    m = metric or Metric()
    results = {}
    violations = {}

    # 1. Four-velocity normalization
    u_lower = m.g @ u
    u_norm_squared = np.dot(u, u_lower)
    expected_norm = -(PhysicalConstants.c**2)
    norm_violation = abs(u_norm_squared - expected_norm)
    results["velocity_normalization"] = norm_violation < tolerance
    violations["velocity_normalization"] = norm_violation

    # 2. Spatial projector idempotency
    h_proj = spatial_projector(u, m)
    h_squared = h_proj @ h_proj
    idempotency_violation = np.max(np.abs(h_squared - h_proj))
    results["spatial_projector_idempotency"] = idempotency_violation < tolerance
    violations["spatial_projector_idempotency"] = idempotency_violation

    # 3. Shear stress tensor constraints (if present)
    if "pi" in state:
        pi = state["pi"]
        if pi.shape == (m.dim, m.dim):
            ok, details = is_symmetric_traceless_spatial(pi, u, m, tolerance)
            results["shear_tensor_symmetric"] = details["symmetric"]
            results["shear_tensor_traceless"] = details["traceless"]
            results["shear_tensor_spatial"] = details["spatial_left"] and details["spatial_right"]

            # Calculate individual violations for shear tensor
            # Symmetry violation
            sym_violation = np.max(np.abs(pi - pi.T))
            violations["shear_tensor_symmetric"] = sym_violation

            # Traceless violation
            g_inv = np.linalg.inv(m.g)
            trace = float(np.einsum("mn,mn->", g_inv, pi))
            violations["shear_tensor_traceless"] = abs(trace)

            # Spatial violation (orthogonality to u)
            u_pi = np.einsum("m,mn->n", u, pi)
            pi_u = np.einsum("mn,n->m", pi, u)
            spatial_violation = max(np.max(np.abs(u_pi)), np.max(np.abs(pi_u)))
            violations["shear_tensor_spatial"] = spatial_violation

    # 4. Heat flux constraints (if present)
    if "q" in state:
        q = state["q"]
        if q.shape == (m.dim,):
            spatial_ok = is_spatial_vector(q, u, m, tolerance)
            results["heat_flux_spatial"] = spatial_ok

            # Calculate heat flux violation
            q_dot_u = float(np.dot(m.g @ u, q))
            violations["heat_flux_spatial"] = abs(q_dot_u)

    # 5. Bulk pressure (should be scalar - no tensor constraints)
    if "Pi" in state:
        Pi = state["Pi"]
        bulk_ok = np.isscalar(Pi) or (isinstance(Pi, np.ndarray) and Pi.shape == ())
        results["bulk_pressure_scalar"] = bulk_ok
        violations["bulk_pressure_scalar"] = 0.0 if bulk_ok else 1.0

    # Generate final report
    all_satisfied = all(results.values())

    return ValidationReport(
        all_satisfied=all_satisfied,
        individual_results=results,
        violation_magnitudes=violations,
        tolerance=tolerance,
        total_constraints=len(results),
    )


def constraint_violation_magnitude(
    state: dict[str, np.ndarray], u: np.ndarray, metric: Metric | None = None
) -> dict[str, float]:
    """Quantify how badly constraints are violated.

    Returns violation magnitudes for all constraints without applying
    tolerance checks. Useful for debugging and optimization.

    Args:
        state: Dictionary containing field values
        u: Four-velocity array
        metric: Metric tensor (defaults to Minkowski)

    Returns:
        Dictionary mapping constraint names to violation magnitudes
    """
    report = validate_all_constraints(state, u, metric, tolerance=0.0)
    return report.violation_magnitudes


def verify_projector_idempotency(h: np.ndarray, tolerance: float = 1e-12) -> tuple[bool, float]:
    """Verify that a projector satisfies h² = h.

    Args:
        h: Projector matrix to verify
        tolerance: Numerical tolerance for idempotency check

    Returns:
        Tuple of (is_idempotent, max_violation)
    """
    h_squared = h @ h
    violation = np.max(np.abs(h_squared - h))
    is_idempotent = violation < tolerance
    return is_idempotent, violation


def test_all_reference_frames(
    state_generator: Callable[[str], tuple[dict[str, Any], np.ndarray]], 
    frames: list[str] = None, 
    tolerance: float = 1e-12
) -> dict[str, ValidationReport]:
    """Test constraint validation across different reference frames.

    Args:
        state_generator: Function that generates (state, u) for each frame
        frames: List of frame names to test (default: rest, moving, boosted)
        tolerance: Numerical tolerance for all tests

    Returns:
        Dictionary mapping frame names to validation reports
    """
    frames = frames or ["rest", "moving", "boosted"]
    results = {}

    for frame in frames:
        try:
            state, u = state_generator(frame)
            report = validate_all_constraints(state, u, tolerance=tolerance)
            results[frame] = report
        except Exception as e:
            # Create error report for failed frame
            results[frame] = ValidationReport(
                all_satisfied=False,
                individual_results={"error": False},
                violation_magnitudes={"error": float("inf")},
                tolerance=tolerance,
                total_constraints=1,
            )

    return results


def apply_constraint_corrections(
    state: dict[str, np.ndarray],
    u: np.ndarray,
    metric: Metric | None = None,
    max_iterations: int = 10,
    tolerance: float = 1e-12,
) -> tuple[dict[str, np.ndarray], ValidationReport]:
    """Apply constraint corrections to enforce physical constraints.

    Iteratively projects fields onto constraint surfaces to satisfy
    physical requirements. Warns if constraints cannot be satisfied.

    Args:
        state: Input field state
        u: Four-velocity (must be normalized)
        metric: Metric tensor
        max_iterations: Maximum correction iterations
        tolerance: Target tolerance for constraint satisfaction

    Returns:
        Tuple of (corrected_state, final_validation_report)
    """
    m = metric or Metric()
    corrected_state = state.copy()

    for _iteration in range(max_iterations):
        # Apply corrections

        # Correct shear tensor if present
        if "pi" in corrected_state:
            pi = corrected_state["pi"]
            if pi.shape == (m.dim, m.dim):
                # Symmetrize
                pi = 0.5 * (pi + pi.T)

                # Make traceless
                g_inv = np.linalg.inv(m.g)
                trace = np.einsum("mn,mn->", g_inv, pi)
                pi = pi - (trace / m.dim) * g_inv

                # Project to spatial subspace (orthogonal to u)
                u_lower = m.g @ u
                for mu in range(m.dim):
                    for nu in range(m.dim):
                        # Remove components parallel to u
                        u_component = (
                            np.sum(pi[mu, :] * u_lower) * u_lower[nu]
                            + np.sum(pi[:, nu] * u) * u_lower[mu]
                        ) / (2 * np.dot(u, u_lower))
                        pi[mu, nu] -= u_component

                corrected_state["pi"] = pi

        # Project heat flux to spatial subspace
        if "q" in corrected_state:
            q = corrected_state["q"]
            if q.shape == (m.dim,):
                u_lower = m.g @ u
                u_norm_sq = np.dot(u, u_lower)
                q_dot_u = np.dot(q, u_lower)
                q_corrected = q - (q_dot_u / u_norm_sq) * u
                corrected_state["q"] = q_corrected

        # Check if constraints are now satisfied
        report = validate_all_constraints(corrected_state, u, m, tolerance)
        if report.all_satisfied:
            break

    else:
        # Maximum iterations reached without full convergence
        import warnings

        warnings.warn(
            f"Constraint correction did not fully converge after {max_iterations} iterations. "
            f"Final constraint violations: {report.violation_magnitudes}",
            stacklevel=2,
        )

    return corrected_state, report
