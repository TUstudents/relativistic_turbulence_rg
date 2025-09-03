"""
Beta Function Calculations for Relativistic Israel-Stewart Theory.

This module implements the renormalization group β-functions for transport coefficients
in relativistic Israel-Stewart hydrodynamics. The β-functions describe how transport
coefficients scale with energy/momentum cutoff in the effective field theory.

Theoretical Background:
    The Israel-Stewart theory has three main transport coefficients:
    - η: Shear viscosity
    - ζ: Bulk viscosity
    - κ: Thermal conductivity

    Their RG flow equations are:
    - β_η = μ dη/dμ = -2η + α_η η²/s + ...
    - β_ζ = μ dζ/dμ = -ζ + α_ζ ζ²/s + ...
    - β_κ = μ dκ/dμ = -κ + α_κ κ²/s + ...

    where s is the entropy density and α coefficients come from loop calculations.

References:
    - Kovtun, P. et al. "AdS/CFT and the entropy bound" Phys. Rev. D 74 (2006)
    - Romatschke, P. "New developments in relativistic viscous hydrodynamics"
"""

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass
class RGFlowParameters:
    """Parameters for RG flow integration."""

    initial_scale: float = 1.0  # Initial energy scale μ₀
    final_scale: float = 0.1  # Final energy scale μ_f
    n_steps: int = 100  # Integration steps
    temperature: float = 1.0  # Background temperature
    entropy_density: float = 1.0  # Background entropy density


class BetaFunctionCalculator:
    """
    Calculate β-functions for Israel-Stewart transport coefficients.

    This class implements the renormalization group equations for the three
    main transport coefficients in relativistic hydrodynamics: shear viscosity η,
    bulk viscosity ζ, and thermal conductivity κ.

    The β-functions capture the scale dependence of these coefficients,
    which is crucial for understanding their universal properties and
    the approach to hydrodynamic scaling.
    """

    def __init__(self, flow_params: RGFlowParameters | None = None):
        """Initialize beta function calculator.

        Args:
            flow_params: Parameters for RG flow integration
        """
        self.flow_params = flow_params or RGFlowParameters()

        # Wilson coefficients (from loop calculations)
        # These would be computed from explicit field theory calculations
        self.alpha_eta = 1.0 / (4 * sp.pi)  # One-loop coefficient for shear viscosity
        self.alpha_zeta = 1.0 / (8 * sp.pi)  # One-loop coefficient for bulk viscosity
        self.alpha_kappa = 1.0 / (6 * sp.pi)  # One-loop coefficient for thermal conductivity

        # Anomalous dimensions (from AdS/CFT and field theory)
        self.gamma_eta = 0.0  # Anomalous dimension for η
        self.gamma_zeta = 1.0  # Anomalous dimension for ζ
        self.gamma_kappa = 1.0  # Anomalous dimension for κ

    def beta_shear_viscosity(self, eta: float, mu: float) -> Any:
        """
        β-function for shear viscosity: β_η = μ dη/dμ.

        The shear viscosity has a special role in relativistic hydrodynamics
        due to the KSS (Kovtun-Son-Starinets) bound η/s ≥ 1/(4π).

        Args:
            eta: Current value of shear viscosity
            mu: Energy scale

        Returns:
            β_η = μ dη/dμ
        """
        s = self.flow_params.entropy_density

        # Leading order: anomalous dimension contribution
        beta_LO = -self.gamma_eta * eta

        # Next-to-leading order: Wilson coefficient contribution
        beta_NLO = self.alpha_eta * eta**2 / s

        # Include temperature dependence for realistic flow
        T = self.flow_params.temperature
        temp_correction = -eta * (mu / T) ** 2 * 1e-3  # Small temperature correction

        return beta_LO + beta_NLO + temp_correction

    def beta_bulk_viscosity(self, zeta: float, mu: float) -> Any:
        """
        β-function for bulk viscosity: β_ζ = μ dζ/dμ.

        Bulk viscosity is associated with conformal symmetry breaking.
        In conformal theories, ζ = 0, but RG flow can generate it.

        Args:
            zeta: Current value of bulk viscosity
            mu: Energy scale

        Returns:
            β_ζ = μ dζ/dμ
        """
        s = self.flow_params.entropy_density

        # Leading order: anomalous dimension (conformal breaking)
        beta_LO = -self.gamma_zeta * zeta

        # Next-to-leading order: Wilson coefficient
        beta_NLO = self.alpha_zeta * zeta**2 / s

        # Conformal breaking scale (bulk viscosity vanishes at conformal point)
        conformal_breaking = -zeta * mu * 1e-2  # Small conformal breaking

        return beta_LO + beta_NLO + conformal_breaking

    def beta_thermal_conductivity(self, kappa: float, mu: float) -> Any:
        """
        β-function for thermal conductivity: β_κ = μ dκ/dμ.

        Thermal conductivity governs heat transport and is related to
        the Wiedemann-Franz law in the quantum critical regime.

        Args:
            kappa: Current value of thermal conductivity
            mu: Energy scale

        Returns:
            β_κ = μ dκ/dμ
        """
        s = self.flow_params.entropy_density
        T = self.flow_params.temperature

        # Leading order: anomalous dimension
        beta_LO = -self.gamma_kappa * kappa

        # Next-to-leading order: Wilson coefficient
        beta_NLO = self.alpha_kappa * kappa**2 / s

        # Wiedemann-Franz relation correction
        # κ/T ∼ σ_el (electrical conductivity) in quantum critical systems
        wiedemann_franz = kappa * (mu / T) * 1e-3

        return beta_LO + beta_NLO + wiedemann_franz

    def compute_full_beta_system(
        self, transport_coeffs: dict[str, float], mu: float
    ) -> dict[str, float]:
        """
        Compute all β-functions simultaneously for coupled RG flow.

        Args:
            transport_coeffs: Current values {"eta": η, "zeta": ζ, "kappa": κ}
            mu: Energy scale

        Returns:
            Dictionary of β-function values
        """
        eta = transport_coeffs.get("eta", 0.1)
        zeta = transport_coeffs.get("zeta", 0.05)
        kappa = transport_coeffs.get("kappa", 0.08)

        beta_functions = {
            "beta_eta": self.beta_shear_viscosity(eta, mu),
            "beta_zeta": self.beta_bulk_viscosity(zeta, mu),
            "beta_kappa": self.beta_thermal_conductivity(kappa, mu),
        }

        # Add coupling effects (transport coefficients can influence each other)
        # η-ζ coupling (shear-bulk coupling)
        coupling_eta_zeta = 0.01 * eta * zeta / self.flow_params.entropy_density
        beta_functions["beta_eta"] += coupling_eta_zeta
        beta_functions["beta_zeta"] += coupling_eta_zeta

        # κ-η coupling (thermal-shear coupling via Onsager relations)
        coupling_kappa_eta = 0.005 * kappa * eta / self.flow_params.entropy_density
        beta_functions["beta_kappa"] += coupling_kappa_eta
        beta_functions["beta_eta"] += coupling_kappa_eta

        return beta_functions

    def integrate_rg_flow(self, initial_coeffs: dict[str, float]) -> dict[str, list[float]]:
        """
        Integrate RG flow equations from initial to final scale.

        Args:
            initial_coeffs: Initial transport coefficients

        Returns:
            Dictionary with scale evolution of each coefficient
        """
        # Logarithmic scale evolution
        mu_initial = self.flow_params.initial_scale
        mu_final = self.flow_params.final_scale
        n_steps = self.flow_params.n_steps

        # Log-scale steps
        log_mu_initial = sp.log(mu_initial)
        log_mu_final = sp.log(mu_final)
        d_log_mu = (log_mu_final - log_mu_initial) / n_steps

        # Initialize arrays
        scales = []
        eta_values = []
        zeta_values = []
        kappa_values = []

        # Current values
        current_coeffs = initial_coeffs.copy()
        current_log_mu = log_mu_initial

        # Euler integration of β-functions
        for step in range(n_steps + 1):
            mu = sp.exp(current_log_mu)
            scales.append(float(mu))
            eta_values.append(current_coeffs["eta"])
            zeta_values.append(current_coeffs["zeta"])
            kappa_values.append(current_coeffs["kappa"])

            if step < n_steps:  # Don't update on final step
                # Compute β-functions
                betas = self.compute_full_beta_system(current_coeffs, float(mu))

                # Update using β-functions: dg/d(ln μ) = β_g
                current_coeffs["eta"] += betas["beta_eta"] * d_log_mu
                current_coeffs["zeta"] += betas["beta_zeta"] * d_log_mu
                current_coeffs["kappa"] += betas["beta_kappa"] * d_log_mu

                current_log_mu += d_log_mu

        return {"scales": scales, "eta": eta_values, "zeta": zeta_values, "kappa": kappa_values}

    def find_fixed_points(self) -> dict[str, float]:
        """
        Find fixed points of the RG flow where all β-functions vanish.

        Returns:
            Dictionary of fixed point values
        """
        # For simplified analysis, solve β = 0 analytically
        s = self.flow_params.entropy_density

        # Fixed points (approximate solutions to β = 0)
        # For η: -γ_η η + α_η η²/s = 0 → η* = γ_η s/α_η
        eta_fixed = self.gamma_eta * s / self.alpha_eta if self.alpha_eta != 0 else 0.0

        # For ζ: -γ_ζ ζ + α_ζ ζ²/s = 0 → ζ* = γ_ζ s/α_ζ
        zeta_fixed = self.gamma_zeta * s / self.alpha_zeta if self.alpha_zeta != 0 else 0.0

        # For κ: -γ_κ κ + α_κ κ²/s = 0 → κ* = γ_κ s/α_κ
        kappa_fixed = self.gamma_kappa * s / self.alpha_kappa if self.alpha_kappa != 0 else 0.0

        return {"eta_fixed": eta_fixed, "zeta_fixed": zeta_fixed, "kappa_fixed": kappa_fixed}

    def compute_critical_exponents(self) -> dict[str, float]:
        """
        Compute critical exponents near fixed points.

        Returns:
            Dictionary of critical exponents
        """
        # Critical exponents are related to eigenvalues of β-function Jacobian
        # at fixed points. For our simplified system:

        critical_exponents = {
            "nu_eta": 1.0 / self.gamma_eta if self.gamma_eta != 0 else float("inf"),
            "nu_zeta": 1.0 / self.gamma_zeta if self.gamma_zeta != 0 else float("inf"),
            "nu_kappa": 1.0 / self.gamma_kappa if self.gamma_kappa != 0 else float("inf"),
        }

        return critical_exponents
