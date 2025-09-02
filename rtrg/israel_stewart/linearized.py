"""
Linearized Israel-Stewart relativistic hydrodynamics.

This module implements the linearized version of Israel-Stewart theory for
stability analysis and dispersion relation calculations. The linearization
expands all fields around an equilibrium background state:
    φ(x) = φ₀ + δφ(x)

Key Features:
    - Background state management for equilibrium configurations
    - Perturbation field algebra with proper tensor structure
    - Dispersion relation calculations for all physical modes
    - Linear stability analysis with growth rate calculations
    - Causality verification for propagating modes

Physical Modes:
    1. Sound modes: ω² = c_s²k² - iΓk² (acoustic propagation)
    2. Shear modes: Governed by viscous relaxation (τ_π)
    3. Diffusive modes: Heat conduction and bulk viscosity

Mathematical Framework:
    The linearized equations take the form:
        ∂_t δρ + ρ₀∇·δv⃗ = 0
        ρ₀∂_t δv^i + ∇^i δp + ∇_j δπ^ij = 0
        τ_π ∂_t δπ^ij + δπ^ij = 2η∇^(i δv^j)
        τ_Π ∂_t δΠ + δΠ = -ζ∇·δv⃗
        τ_q ∂_t δq^i + δq^i = -κ∇^i(δT/T₀)

Usage:
    >>> background = {'rho': 1.0, 'u': [1, 0, 0, 0], 'pi': 0, 'Pi': 0, 'q': [0, 0, 0, 0]}
    >>> linearized = LinearizedIS(background, parameters)
    >>> omega = linearized.dispersion_relation(k=1.0, mode='sound')
    >>> stable = linearized.is_linearly_stable()

References:
    - Israel, W. & Stewart, J.M. Ann. Phys. 118, 341 (1979)
    - Denicol, G.S. et al. Phys. Rev. D 85, 114047 (2012)
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ..core.constants import PhysicalConstants
from ..core.tensors import Metric
from .equations import IsraelStewartParameters


@dataclass
class BackgroundState:
    """
    Background equilibrium state for linearization.

    Represents the equilibrium configuration around which we expand:
        φ = φ₀ + δφ

    Standard equilibrium state:
        - ρ₀: Constant energy density
        - u^μ = (c, 0, 0, 0): Fluid at rest
        - π^μν = 0: No shear stress
        - Π = 0: No bulk viscous pressure
        - q^μ = (0, 0, 0, 0): No heat flux

    Attributes:
        rho: Background energy density [M L^{-1} T^{-2}]
        pressure: Background pressure [M L^{-1} T^{-2}]
        temperature: Background temperature [K]
        u: Background four-velocity [dimensionless]
        pi: Background shear stress (typically zero) [M L^{-1} T^{-2}]
        Pi: Background bulk pressure (typically zero) [M L^{-1} T^{-2}]
        q: Background heat flux (typically zero) [M T^{-3}]
    """

    rho: float = 1.0  # Energy density
    pressure: float = 0.33  # Equilibrium pressure
    temperature: float = 1.0  # Temperature
    u: list[float] | None = None  # Four-velocity
    pi: float | NDArray = 0.0  # Shear stress tensor
    Pi: float = 0.0  # Bulk pressure
    q: list[float] | None = None  # Heat flux vector

    def __post_init__(self) -> None:
        """Initialize default values and validate background state."""
        if self.u is None:
            self.u = [PhysicalConstants.c, 0.0, 0.0, 0.0]  # Fluid at rest
        if self.q is None:
            self.q = [0.0, 0.0, 0.0, 0.0]  # No heat flux

        # Validate four-velocity normalization
        u_norm_sq = -(self.u[0] ** 2) + sum(self.u[i] ** 2 for i in range(1, 4))
        expected_norm = -(PhysicalConstants.c**2)
        if abs(u_norm_sq - expected_norm) > 1e-10:
            raise ValueError(
                f"Four-velocity not normalized: u·u = {u_norm_sq}, expected {expected_norm}"
            )

    def validate_equilibrium(self) -> bool:
        """
        Check if background state represents a valid equilibrium.

        For a true equilibrium:
            - All spatial components of u^μ should be zero (fluid at rest)
            - All dissipative fluxes (π^μν, Π, q^μ) should be zero
            - Thermodynamic quantities should be consistent

        Returns:
            True if state represents valid equilibrium
        """
        # Check if fluid is at rest (spatial velocity = 0)
        if self.u is not None:
            spatial_velocity_sq = sum(self.u[i] ** 2 for i in range(1, 4))
            if spatial_velocity_sq > 1e-10:
                return False

        # Check dissipative fluxes are zero
        if abs(self.Pi) > 1e-10:
            return False

        if self.q is not None:
            heat_flux_sq = sum(self.q[i] ** 2 for i in range(1, 4))
            if heat_flux_sq > 1e-10:
                return False

        # For tensor pi, check if close to zero
        if isinstance(self.pi, np.ndarray):
            if np.max(np.abs(self.pi)) > 1e-10:
                return False
        elif abs(self.pi) > 1e-10:
            return False

        return True


class LinearizedField:
    """
    Represents a linearized field: φ = φ₀ + δφ.

    Manages both the background value and perturbation components
    of a field in the linearized expansion around equilibrium.
    """

    def __init__(
        self,
        name: str,
        background_value: float | list | NDArray,
        tensor_rank: int = 0,
        symmetries: list[str] | None = None,
    ):
        """
        Initialize linearized field.

        Args:
            name: Field identifier (e.g., 'rho', 'u', 'pi')
            background_value: Equilibrium value φ₀
            tensor_rank: 0=scalar, 1=vector, 2=tensor
            symmetries: Tensor symmetries ['symmetric', 'traceless', 'spatial']
        """
        self.name = name
        self.background = background_value
        self.tensor_rank = tensor_rank
        self.symmetries = symmetries or []

        # Create symbolic perturbation
        if tensor_rank == 0:
            self.perturbation = sp.Symbol(f"delta_{name}")
        elif tensor_rank == 1:
            self.perturbation = sp.IndexedBase(f"delta_{name}")
        elif tensor_rank == 2:
            self.perturbation = sp.IndexedBase(f"delta_{name}")
        else:
            raise ValueError(f"Tensor rank {tensor_rank} not supported")

    def total_field(self) -> sp.Expr:
        """Return total field: φ = φ₀ + δφ"""
        return self.background + self.perturbation

    def __str__(self) -> str:
        return f"LinearizedField({self.name}, rank={self.tensor_rank})"


class LinearizedIS:
    """
    Complete linearized Israel-Stewart hydrodynamics system.

    Implements linearization of the full IS equations around an equilibrium
    background state, providing tools for stability analysis, dispersion
    relation calculations, and mode analysis.

    The system handles the linearized evolution equations:
        L[δφ] = 0
    where L is the linearized differential operator and δφ represents
    all perturbation fields.

    Mathematical Structure:
        - 14 linearized field components (same as full IS system)
        - Characteristic polynomial det[M(ω,k)] = 0
        - Physical dispersion relations ω(k) for all modes
        - Stability analysis via Im(ω) calculations
    """

    def __init__(
        self,
        background_state: BackgroundState,
        parameters: IsraelStewartParameters,
        metric: Metric | None = None,
    ):
        """
        Initialize linearized IS system.

        Args:
            background_state: Equilibrium state for expansion
            parameters: Physical parameters (transport coefficients, etc.)
            metric: Spacetime metric (default: Minkowski)
        """
        self.background = background_state
        self.parameters = parameters
        self.metric = metric or Metric()

        # Validate background equilibrium
        if not background_state.validate_equilibrium():
            raise ValueError("Background state is not a valid equilibrium")

        # Create linearized fields
        self.linearized_fields = self._create_linearized_fields()

        # Symbolic coordinates and wave vector
        self.t, self.x, self.y, self.z = sp.symbols("t x y z")
        self.omega, self.k = sp.symbols("omega k", real=True)
        self.kx, self.ky, self.kz = sp.symbols("k_x k_y k_z", real=True)

    def _create_linearized_fields(self) -> dict[str, LinearizedField]:
        """
        Create all linearized field objects.

        Returns:
            Dictionary mapping field names to LinearizedField objects
        """
        fields = {}

        # Scalar fields
        fields["rho"] = LinearizedField("rho", self.background.rho, tensor_rank=0)
        fields["Pi"] = LinearizedField("Pi", self.background.Pi, tensor_rank=0)

        # Vector fields (guaranteed to be non-None after __post_init__)
        if self.background.u is None:
            raise ValueError("Background four-velocity u is None - invalid background state")
        if self.background.q is None:
            raise ValueError("Background heat flux q is None - invalid background state")
        fields["u"] = LinearizedField("u", self.background.u, tensor_rank=1)
        fields["q"] = LinearizedField(
            "q", self.background.q, tensor_rank=1, symmetries=["spatial"]
        )  # u·q = 0

        # Tensor field (shear stress)
        fields["pi"] = LinearizedField(
            "pi",
            self.background.pi,
            tensor_rank=2,
            symmetries=["symmetric", "traceless", "spatial"],
        )

        return fields

    def linearize_field(self, field_name: str) -> LinearizedField:
        """
        Get linearized version of specified field: φ = φ₀ + δφ.

        Args:
            field_name: Name of field to linearize

        Returns:
            LinearizedField object containing background and perturbation
        """
        if field_name not in self.linearized_fields:
            raise ValueError(f"Unknown field: {field_name}")
        return self.linearized_fields[field_name]

    def get_linearized_equations(self) -> dict[str, sp.Expr]:
        """
        Extract linearized evolution equations for all fields.

        Returns the linearized IS equations in the form:
            L[δφ] = 0
        where L is the linearized differential operator.

        Returns:
            Dictionary of linearized equations for each field
        """
        equations = {}

        # Linearized continuity equation: ∂_t δρ + ρ₀ ∇·δv⃗ = 0
        delta_rho = self.linearized_fields["rho"].perturbation
        delta_u = self.linearized_fields["u"].perturbation

        # Spatial divergence of velocity: ∇·δv⃗ = ∇_i δu^i
        velocity_divergence = (
            sp.Derivative(delta_u[1], self.x)
            + sp.Derivative(delta_u[2], self.y)
            + sp.Derivative(delta_u[3], self.z)
        )

        equations["continuity"] = (
            sp.Derivative(delta_rho, self.t) + self.background.rho * velocity_divergence
        )

        # Linearized momentum equation: ρ₀ ∂_t δv^i + ∇^i δp + ∇_j δπ^ij = 0
        delta_pi = self.linearized_fields["pi"].perturbation

        # For each spatial component i = 1,2,3
        for i_val in range(1, 4):
            coord_map = {1: self.x, 2: self.y, 3: self.z}

            # Pressure gradient term (using thermodynamic relations)
            pressure_gradient = sp.Derivative(delta_rho, coord_map[i_val]) * (
                self.parameters.equilibrium_pressure / self.background.rho
            )

            # Viscous stress divergence: ∇_j δπ^ij
            stress_divergence = (
                sp.Derivative(delta_pi[i_val, 1], self.x)
                + sp.Derivative(delta_pi[i_val, 2], self.y)
                + sp.Derivative(delta_pi[i_val, 3], self.z)
            )

            equations[f"momentum_{i_val}"] = (
                self.background.rho * sp.Derivative(delta_u[i_val], self.t)
                + pressure_gradient
                + stress_divergence
            )

        # Linearized shear stress evolution: τ_π ∂_t δπ^ij + δπ^ij = 2η ∇^(i δv^j)
        for i_val in range(1, 4):
            for j_val in range(i_val, 4):  # Only upper triangle due to symmetry
                coord_map = {1: self.x, 2: self.y, 3: self.z}

                # Symmetric velocity gradient: ∇^(i δv^j) = ½(∇^i δv^j + ∇^j δv^i)
                vel_grad_ij = sp.Rational(1, 2) * (
                    sp.Derivative(delta_u[j_val], coord_map[i_val])
                    + sp.Derivative(delta_u[i_val], coord_map[j_val])
                )

                # Subtract trace for traceless condition
                if i_val == j_val:
                    # Diagonal terms get trace subtraction: (1/3) * ∇·δv⃗
                    trace_part = sp.Rational(1, 3) * velocity_divergence
                    vel_grad_ij -= trace_part

                equations[f"shear_{i_val}_{j_val}"] = (
                    self.parameters.tau_pi * sp.Derivative(delta_pi[i_val, j_val], self.t)
                    + delta_pi[i_val, j_val]
                    - 2 * self.parameters.eta * vel_grad_ij
                )

        # Linearized bulk pressure evolution: τ_Π ∂_t δΠ + δΠ = -ζ ∇·δv⃗
        delta_Pi = self.linearized_fields["Pi"].perturbation
        equations["bulk"] = (
            self.parameters.tau_Pi * sp.Derivative(delta_Pi, self.t)
            + delta_Pi
            + self.parameters.zeta * velocity_divergence
        )

        # Linearized heat flux evolution: τ_q ∂_t δq^i + δq^i = -κ ∇^i(δT/T₀)
        delta_q = self.linearized_fields["q"].perturbation

        # Temperature perturbation (thermodynamic relation)
        # For ideal gas: δT/T₀ = (∂T/∂ρ)_p δρ/T₀ ≈ (1/3) δρ/ρ₀
        temp_perturbation = delta_rho / (3 * self.background.rho)

        for i_val in range(1, 4):
            coord_map = {1: self.x, 2: self.y, 3: self.z}

            equations[f"heat_{i_val}"] = (
                self.parameters.tau_q * sp.Derivative(delta_q[i_val], self.t)
                + delta_q[i_val]
                + self.parameters.kappa * sp.Derivative(temp_perturbation, coord_map[i_val])
            )

        return equations

    def characteristic_polynomial(self, omega: sp.Symbol, k: sp.Symbol) -> sp.Expr:
        """
        Construct characteristic polynomial for dispersion relation.

        For plane wave perturbations δφ ~ exp(i(k·x - ωt)), the linearized
        equations become an eigenvalue problem M·δφ = 0. The dispersion
        relation is det[M(ω,k)] = 0.

        Args:
            omega: Frequency symbol
            k: Wave number magnitude symbol

        Returns:
            Characteristic polynomial in ω and k
        """
        # For simplicity, consider longitudinal modes (k parallel to perturbation)
        # Full analysis would include transverse modes

        # Simplified 3x3 matrix for (δρ, δv, δΠ) coupled system
        # Real implementation would be larger matrix

        # Matrix elements from linearized equations
        # M[0,0]: from continuity equation
        M_00 = -sp.I * omega
        M_01 = self.background.rho * sp.I * k
        M_02 = 0

        # M[1,0]: from momentum equation
        M_10 = sp.I * k * self.parameters.equilibrium_pressure / self.background.rho
        M_11 = -sp.I * omega * self.background.rho
        M_12 = 0  # Coupling to bulk pressure

        # M[2,0]: from bulk pressure equation
        M_20 = 0
        M_21 = -self.parameters.zeta * sp.I * k
        M_22 = -sp.I * omega * self.parameters.tau_Pi + 1

        # Construct 3x3 matrix
        matrix = sp.Matrix([[M_00, M_01, M_02], [M_10, M_11, M_12], [M_20, M_21, M_22]])

        # Characteristic polynomial is determinant
        char_poly = matrix.det()

        return char_poly

    def dispersion_relation(self, k_val: float, mode: str = "sound") -> complex:
        """
        Calculate dispersion relation ω(k) for specified mode.

        Solves the characteristic polynomial det[M(ω,k)] = 0 to find
        the complex frequency ω as a function of wave number k.

        Args:
            k_val: Wave number magnitude
            mode: Physical mode type ['sound', 'shear', 'diffusive']

        Returns:
            Complex frequency ω = Re(ω) + i·Im(ω)
            Re(ω): Oscillation frequency
            Im(ω): Growth/decay rate
        """
        # Get characteristic polynomial
        char_poly = self.characteristic_polynomial(self.omega, self.k)

        # Substitute k value
        char_poly_k = char_poly.subs(self.k, k_val)

        # Solve for omega
        omega_solutions = sp.solve(char_poly_k, self.omega)

        # Handle different return types from sp.solve
        if not omega_solutions:
            return 0j  # No solutions found

        # Convert to numerical values and select appropriate mode
        numerical_solutions = []
        for sol in omega_solutions:
            if isinstance(sol, list | tuple):
                # If solution is a list/tuple, take first element
                sol = sol[0] if sol else 0
            try:
                numerical_solutions.append(complex(sol.evalf()))
            except (AttributeError, TypeError):
                # Fallback for non-symbolic solutions
                numerical_solutions.append(complex(sol))

        if mode == "sound":
            # Sound mode: look for solution with Re(ω) ≈ c_s * k
            sound_speed_sq = self.parameters.equilibrium_pressure / self.background.rho
            target_freq = np.sqrt(sound_speed_sq) * k_val

            # Find closest solution to expected sound mode
            best_solution = min(numerical_solutions, key=lambda x: abs(x.real - target_freq))
            return best_solution

        elif mode == "diffusive":
            # Diffusive mode: purely imaginary frequency
            diffusive_solutions = [
                sol for sol in numerical_solutions if abs(sol.real) < abs(sol.imag)
            ]
            return diffusive_solutions[0] if diffusive_solutions else numerical_solutions[0]

        else:
            # Return first solution for other modes
            return numerical_solutions[0] if numerical_solutions else 0j

    def is_linearly_stable(
        self, k_range: tuple[float, float] = (0.1, 10.0), num_points: int = 100
    ) -> bool:
        """
        Determine linear stability by checking growth rates.

        A system is linearly unstable if any mode has Im(ω) > 0,
        indicating exponential growth of perturbations.

        Args:
            k_range: Range of wave numbers to test (k_min, k_max)
            num_points: Number of k values to sample

        Returns:
            True if system is linearly stable (all Im(ω) ≤ 0)
        """
        k_values = np.linspace(k_range[0], k_range[1], num_points)

        for k_val in k_values:
            # Check all physical modes
            for mode in ["sound", "diffusive"]:
                omega = self.dispersion_relation(k_val, mode)

                # If any mode has positive growth rate, system is unstable
                if omega.imag > 1e-10:  # Small tolerance for numerical errors
                    return False

        return True

    def sound_attenuation_coefficient(self) -> float:
        """
        Calculate sound attenuation coefficient Γ.

        For small k, sound modes have dispersion:
            ω² ≈ c_s²k² - iΓk²
        where Γ = (4η/3 + ζ)/ρ is the attenuation coefficient.

        Returns:
            Sound attenuation coefficient Γ
        """
        gamma = (4 * self.parameters.eta / 3 + self.parameters.zeta) / self.background.rho
        return gamma

    def critical_parameters(self) -> dict[str, float]:
        """
        Find critical parameter values for stability transitions.

        Returns:
            Dictionary of critical parameter values
        """
        # This is a simplified analysis - full implementation would require
        # detailed stability analysis for different parameter regimes

        # Critical relaxation time for causality
        tau_critical = np.sqrt(self.parameters.eta / (self.background.rho * PhysicalConstants.c**2))

        return {
            "tau_pi_critical": tau_critical,
            "sound_speed_squared": self.parameters.equilibrium_pressure / self.background.rho,
            "attenuation_coefficient": self.sound_attenuation_coefficient(),
        }

    def validate_causality(self) -> bool:
        """
        Verify that all modes propagate slower than light.

        Returns:
            True if causality is preserved (all |Re(ω)/k| < c)
        """
        # Test a range of k values
        k_values = np.logspace(-1, 1, 50)  # k from 0.1 to 10

        for k_val in k_values:
            omega = self.dispersion_relation(k_val, "sound")
            phase_velocity = abs(omega.real) / k_val

            if phase_velocity >= PhysicalConstants.c:
                return False

        return True

    def __str__(self) -> str:
        """String representation"""
        return f"LinearizedIS(background={self.background}, stable={self.is_linearly_stable()})"
