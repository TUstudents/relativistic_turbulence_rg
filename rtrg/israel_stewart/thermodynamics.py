"""
Thermodynamics and equations of state (EOS) for Israel–Stewart hydrodynamics.

This module provides a minimal, extensible EOS interface and two concrete
implementations used in tests and linearized closures:
    - ConformalEOS: p = ρ/3, c_s^2 = 1/3
    - IdealGasEOS: p = (γ-1) ρ, c_s^2 = γ-1 (constant γ)

All quantities are expressed in natural units. The temperature model is a
simple placeholder sufficient for linearized scaling; detailed thermo can be
introduced later without breaking the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EOS(Protocol):
    """Equation of State protocol.

    An EOS must provide pressure and sound speed squared as functions of energy
    density. Temperature is optional and may be model-dependent.
    """

    def pressure(self, rho: float) -> float:  # pragma: no cover - interface
        ...

    def cs2(self, rho: float) -> float:  # pragma: no cover - interface
        ...

    def temperature(self, rho: float) -> float:  # pragma: no cover - interface
        ...


@dataclass
class ConformalEOS:
    """Conformal fluid EOS with p = ρ/3 and c_s^2 = 1/3.

    Temperature scaling is modeled as T ∝ ρ^{1/4} with proportionality constant
    chosen to be 1 for simplicity (natural units). This is sufficient for linear
    closures in tests and examples.
    """

    def pressure(self, rho: float) -> float:
        if rho < 0:
            raise ValueError("energy density must be non-negative")
        return rho / 3.0

    def cs2(self, rho: float) -> float:  # noqa: ARG002 - rho unused (constant)
        return 1.0 / 3.0

    def temperature(self, rho: float) -> float:
        if rho < 0:
            raise ValueError("energy density must be non-negative")
        # T ~ rho^{1/4} in conformal fluids (up to constants)
        return float(np.power(rho + 1e-30, 0.25))


@dataclass
class IdealGasEOS:
    """Relativistic ideal-gas-like EOS with constant adiabatic index γ.

    p = (γ - 1) ρ,   c_s^2 = γ - 1

    This simplified model is useful for linear closures and sanity checks.
    """

    gamma: float = 4.0 / 3.0
    T0: float = 1.0  # reference temperature at rho0
    rho0: float = 1.0  # reference energy density

    def __post_init__(self) -> None:
        if not (1.0 < self.gamma <= 2.0):
            raise ValueError("gamma should be in (1, 2] for causal fluids")

    def pressure(self, rho: float) -> float:
        if rho < 0:
            raise ValueError("energy density must be non-negative")
        return (self.gamma - 1.0) * rho

    def cs2(self, rho: float) -> float:  # noqa: ARG002 - rho unused (constant)
        # c_s^2 = ∂p/∂ρ = γ - 1
        return self.gamma - 1.0

    def temperature(self, rho: float) -> float:
        if rho < 0:
            raise ValueError("energy density must be non-negative")
        # Very simple scaling law: T ∝ (p/ρ0)^{(γ-1)/γ}; keep proportional to T0
        # In linearized contexts, proportionality is sufficient.
        p = self.pressure(rho)
        p0 = self.pressure(self.rho0)
        # Avoid zero-div for rho=0; keep smooth behavior near 0
        scale = (p / (p0 + 1e-30)) ** ((self.gamma - 1.0) / self.gamma)
        return float(self.T0 * scale)
