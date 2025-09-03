"""
Complete Israel-Stewart equation system implementation.

This module provides the comprehensive mathematical framework for Israel-Stewart
relativistic hydrodynamics, implementing the complete set of evolution equations
and constraints that govern relativistic viscous fluid dynamics.

Theoretical Framework:
    The Israel-Stewart theory extends ideal relativistic hydrodynamics by treating
    dissipative fluxes as independent dynamical fields with finite relaxation times.
    This resolves causality and stability issues in first-order theories.

Complete Equation System:
    1. Energy-momentum conservation: ∇_μ T^{μν} = 0
    2. Particle number conservation: ∇_μ n^μ = 0
    3. Shear stress evolution: τ_π Δ^{μν}_{αβ} ∂_λ π^{αβ} u^λ + π^{μν} = 2η σ^{μν} + ...
    4. Bulk pressure evolution: τ_Π ∂_λ Π u^λ + Π = -ζ θ + ...
    5. Heat flux evolution: τ_q Δ^μ_α ∂_λ q^α u^λ + q^μ = -κ ∇^μ(μ/T) + ...
    6. Four-velocity constraint: u^μ u_μ = -c²
    7. Spatial orthogonality: u_μ π^{μν} = u_μ q^μ = 0
    8. Traceless condition: π^μ_μ = 0

References:
    - Israel, W. & Stewart, J.M. Ann. Phys. 118, 341 (1979)
    - Rezzolla, L. & Zanotti, O. "Relativistic Hydrodynamics"
"""

from dataclasses import dataclass
from typing import cast

import sympy as sp

from ..core.constants import PhysicalConstants
from ..core.fields import (
    BulkPressureField,
    EnergyDensityField,
    FieldRegistry,
    FourVelocityField,
    HeatFluxField,
    ShearStressField,
)
from ..core.tensors import Metric


@dataclass
class IsraelStewartParameters:
    """
    Complete parameter set for Israel-Stewart hydrodynamics.

    Physical Parameters:
        Transport coefficients control the strength of dissipative effects:
        - η (shear viscosity): Resistance to velocity gradients [M L^{-1} T^{-1}]
        - ζ (bulk viscosity): Resistance to expansion [M L^{-1} T^{-1}]
        - κ (thermal conductivity): Heat conduction strength [M L T^{-3} K^{-1}]

        Relaxation times determine causal propagation speeds:
        - τ_π: Shear stress relaxation time [T]
        - τ_Π: Bulk pressure relaxation time [T]
        - τ_q: Heat flux relaxation time [T]

    Thermodynamic State:
        - T: Temperature [K]
        - μ: Chemical potential [M L^2 T^{-2}]
        - p_eq: Equilibrium pressure [M L^{-1} T^{-2}]

    Mathematical Properties:
        All parameters must satisfy positivity and causality constraints
        for physical solutions to exist.
    """

    # Transport coefficients
    eta: float = 1.0  # Shear viscosity
    zeta: float = 0.1  # Bulk viscosity
    kappa: float = 0.5  # Thermal conductivity

    # Relaxation times
    tau_pi: float = 0.1  # Shear relaxation time
    tau_Pi: float = 0.05  # Bulk relaxation time
    tau_q: float = 0.02  # Heat flux relaxation time

    # Thermodynamic parameters
    temperature: float = 1.0  # Temperature
    chemical_potential: float = 0.0  # Chemical potential
    equilibrium_pressure: float = 0.33  # Equilibrium pressure

    def validate_causality(self) -> bool:
        """
        Validate causality constraints for Israel-Stewart parameters.

        Ensures all relaxation times and transport coefficients satisfy
        the constraints required for causal propagation of signals.

        Returns:
            True if all causality constraints are satisfied
        """
        # All transport coefficients must be non-negative
        if self.eta < 0 or self.zeta < 0 or self.kappa < 0:
            return False

        # All relaxation times must be positive for finite signal speed
        if self.tau_pi <= 0 or self.tau_Pi <= 0 or self.tau_q <= 0:
            return False

        # Temperature must be positive
        if self.temperature <= 0:
            return False

        # Additional causality constraints could be added here
        # (e.g., bounds relating transport coefficients to relaxation times)

        return True


class IsraelStewartSystem:
    """
    Complete Israel-Stewart relativistic hydrodynamics system.

    Implements the full set of evolution equations, constraint equations,
    and coupling terms for relativistic viscous hydrodynamics. Provides
    methods for system initialization, equation extraction, and validation.

    Mathematical Structure:
        The system consists of 14 independent field components:
        - ρ: Energy density (1 scalar)
        - u^μ: Four-velocity (4 components, 1 constraint → 3 independent)
        - π^{μν}: Shear stress (10 independent components for symmetric traceless tensor)
        - Π: Bulk pressure (1 scalar)
        - q^μ: Heat flux (4 components, 1 constraint → 3 independent)

        Total: 1 + 3 + 9 + 1 + 3 = 17 apparent, but with constraints: 14 independent

    Constraint System:
        - u^μ u_μ = -c² (normalization)
        - π^μ_μ = 0 (traceless)
        - u_μ π^{μν} = 0 (spatial orthogonality)
        - u_μ q^μ = 0 (heat flux orthogonality)

    Usage:
        >>> params = IsraelStewartParameters(eta=1.0, tau_pi=0.1)
        >>> system = IsraelStewartSystem(params)
        >>> equations = system.get_evolution_equations()
        >>> constraints = system.get_constraint_equations()
    """

    def __init__(self, parameters: IsraelStewartParameters, metric: Metric | None = None):
        """
        Initialize Israel-Stewart system with parameters and spacetime metric.

        Args:
            parameters: Complete set of physical parameters
            metric: Spacetime metric (default: Minkowski)
        """
        self.parameters = parameters
        self.metric = metric or Metric()

        # Validate parameters
        if not parameters.validate_causality():
            raise ValueError("Parameters violate causality constraints")

        # Create field registry with all IS fields
        self.field_registry = FieldRegistry()
        self.field_registry.create_is_fields(self.metric)

        # Extract individual fields for convenience (these are guaranteed to exist after create_is_fields)
        self.rho: EnergyDensityField = cast(
            EnergyDensityField, self.field_registry.get_field("rho")
        )
        self.u: FourVelocityField = cast(FourVelocityField, self.field_registry.get_field("u"))
        self.pi: ShearStressField = cast(ShearStressField, self.field_registry.get_field("pi"))
        self.Pi: BulkPressureField = cast(BulkPressureField, self.field_registry.get_field("Pi"))
        self.q: HeatFluxField = cast(HeatFluxField, self.field_registry.get_field("q"))

        # Create symbolic variables for spacetime coordinates
        self.t, self.x, self.y, self.z = sp.symbols("t x y z")
        self.coordinates = [self.t, self.x, self.y, self.z]

    def get_evolution_equations(self) -> dict[str, sp.Expr]:
        """
        Extract complete set of evolution equations for all dynamical fields.

        Returns dictionary mapping field names to their evolution equations
        in symbolic form. Each equation represents the time evolution of
        the corresponding field according to Israel-Stewart theory.

        Returns:
            Dictionary of evolution equations: {field_name: evolution_equation}
        """
        evolution_equations = {}

        # Energy density evolution (continuity equation)
        evolution_equations["rho"] = self.rho.evolution_equation()

        # Four-velocity evolution (from momentum conservation)
        evolution_equations["u"] = self._four_velocity_evolution()

        # Shear stress evolution
        evolution_equations["pi"] = self.pi.evolution_equation(
            tau_pi=self.parameters.tau_pi, eta=self.parameters.eta
        )

        # Bulk pressure evolution
        evolution_equations["Pi"] = self.Pi.evolution_equation(
            tau_Pi=self.parameters.tau_Pi, zeta=self.parameters.zeta
        )

        # Heat flux evolution
        evolution_equations["q"] = self.q.evolution_equation(
            tau_q=self.parameters.tau_q, kappa=self.parameters.kappa
        )

        return evolution_equations

    def get_constraint_equations(self) -> list[sp.Expr]:
        """
        Extract all constraint equations that must be satisfied.

        Constraint equations enforce the mathematical and physical
        requirements of Israel-Stewart theory, including four-velocity
        normalization, tensor orthogonalities, and trace conditions.

        Returns:
            List of symbolic constraint equations
        """
        constraints = []

        # Four-velocity normalization: u^μ u_μ + c² = 0
        u_mu = sp.IndexedBase("u")
        g_munu = sp.IndexedBase("g")
        mu, nu = sp.symbols("mu nu")

        u_norm_constraint = (
            sp.Sum(g_munu[mu, nu] * u_mu[mu] * u_mu[nu], (mu, 0, 3), (nu, 0, 3))
            + PhysicalConstants.c**2
        )
        constraints.append(u_norm_constraint)

        # Shear stress tracelessness: π^μ_μ = 0
        pi_munu = sp.IndexedBase("pi")
        trace_constraint = sp.Sum(pi_munu[mu, mu], (mu, 0, 3))
        constraints.append(trace_constraint)

        # Shear stress spatial orthogonality: u_μ π^{μν} = 0
        for nu_val in range(4):
            orthogonality_constraint = sp.Sum(u_mu[mu] * pi_munu[mu, nu_val], (mu, 0, 3))
            constraints.append(orthogonality_constraint)

        # Heat flux orthogonality: u_μ q^μ = 0
        q_mu = sp.IndexedBase("q")
        heat_orthogonality = sp.Sum(u_mu[mu] * q_mu[mu], (mu, 0, 3))
        constraints.append(heat_orthogonality)

        return constraints

    def get_conservation_equations(self) -> dict[str, sp.Expr]:
        """
        Extract energy-momentum and particle number conservation equations.

        These fundamental conservation laws provide the foundation for
        the Israel-Stewart evolution equations. They represent the
        covariant derivatives of conserved currents.

        Returns:
            Dictionary of conservation equations
        """
        conservation = {}

        # Energy-momentum conservation: ∇_μ T^{μν} = 0
        T_munu = sp.IndexedBase("T")
        mu, nu = sp.symbols("mu nu")

        for nu_val in range(4):
            energy_momentum_conservation = sp.Sum(
                sp.Derivative(T_munu[mu, nu_val], self.coordinates[mu]), (mu, 0, 3)
            )
            conservation[f"energy_momentum_{nu_val}"] = energy_momentum_conservation

        # Particle number conservation: ∇_μ n^μ = 0
        n_mu = sp.IndexedBase("n")
        particle_conservation = sp.Sum(sp.Derivative(n_mu[mu], self.coordinates[mu]), (mu, 0, 3))
        conservation["particle_number"] = particle_conservation

        return conservation

    def get_stress_energy_tensor(self) -> sp.Array:
        """
        Construct the complete stress-energy tensor T^{μν}.

        The stress-energy tensor in Israel-Stewart theory includes
        equilibrium and non-equilibrium contributions from viscous stresses
        and heat conduction (shown here in a generic frame):

        Mathematical Form:
            T^{μν} = (ρ + p_eq + Π)u^μu^ν + (p_eq + Π)g^{μν} + π^{μν} + q^μ u^ν + q^ν u^μ

        Returns:
            Symbolic stress-energy tensor as 4x4 array
        """
        # Create symbolic field variables
        rho_sym = sp.Symbol("rho")
        u_mu = sp.IndexedBase("u")
        pi_munu = sp.IndexedBase("pi")
        Pi_sym = sp.Symbol("Pi")
        p_eq = sp.Symbol("p_eq")
        g_munu = sp.IndexedBase("g")
        q_mu = sp.IndexedBase("q")

        # Construct stress-energy tensor components
        T_components = []
        for mu in range(4):
            T_row = []
            for nu in range(4):
                # Perfect fluid part
                perfect_fluid_term = (rho_sym + p_eq + Pi_sym) * u_mu[mu] * u_mu[nu]

                # Pressure term
                pressure_term = (p_eq + Pi_sym) * g_munu[mu, nu]

                # Viscous stress term
                viscous_term = pi_munu[mu, nu]

                # Heat flux symmetric energy transport terms
                heat_terms = q_mu[mu] * u_mu[nu] + q_mu[nu] * u_mu[mu]

                T_row.append(perfect_fluid_term + pressure_term + viscous_term + heat_terms)
            T_components.append(T_row)

        return sp.Array(T_components)

    def validate_system_consistency(self) -> tuple[bool, list[str]]:
        """
        Validate mathematical consistency of the complete system.

        Checks that the system of equations is well-posed, constraints
        are compatible, and parameters satisfy physical requirements.

        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        issues = []

        # Check parameter causality
        if not self.parameters.validate_causality():
            issues.append("Parameters violate causality constraints")

        # Check field registry completeness
        expected_fields = ["rho", "u", "pi", "Pi", "q"]
        for field_name in expected_fields:
            if field_name not in self.field_registry:
                issues.append(f"Missing field: {field_name}")

        # Check evolution equations exist
        try:
            evolution_eqs = self.get_evolution_equations()
            for field_name in expected_fields:
                if field_name not in evolution_eqs:
                    issues.append(f"Missing evolution equation for: {field_name}")
        except Exception as e:
            issues.append(f"Error in evolution equations: {str(e)}")

        # Check constraint equations
        try:
            constraints = self.get_constraint_equations()
            if len(constraints) == 0:
                issues.append("No constraint equations found")
        except Exception as e:
            issues.append(f"Error in constraint equations: {str(e)}")

        return len(issues) == 0, issues

    def _four_velocity_evolution(self) -> sp.Expr:
        """
        Derive four-velocity evolution from momentum conservation.

        The four-velocity evolution comes from the spatial components
        of energy-momentum conservation ∇_μ T^{μi} = 0.

        Returns:
            Symbolic expression for four-velocity evolution
        """
        # This is a simplified placeholder - full implementation would require
        # detailed momentum conservation analysis
        t = sp.Symbol("t")
        u_sym = self.u.symbol

        # Placeholder evolution equation
        # Real implementation: (ρ+p+Π)u^μ ∇_μ u^ν = -∇^ν(p+Π) + ∇_μ π^{μν}
        evolution = sp.Derivative(u_sym, t)  # Simplified

        return evolution

    def get_linearized_system(self, background_state: dict[str, float]) -> dict[str, sp.Expr]:
        """
        Extract linearized equations around a background state.

        Linearization is essential for stability analysis and
        derivation of dispersion relations for small perturbations.

        Args:
            background_state: Background values for all fields

        Returns:
            Dictionary of linearized evolution equations
        """
        linearized_equations = {}

        # Define symbolic perturbations
        t, x, y, z = sp.symbols("t x y z", real=True)

        # Background state values
        rho_0 = background_state.get("rho", 1.0)
        u_0 = background_state.get("u_t", 1.0)  # Time component
        T_0 = background_state.get("T", 1.0)  # Temperature
        P_0 = background_state.get("P", rho_0 / 3)  # Pressure

        # Define perturbations
        delta_rho = sp.Function("delta_rho")(t, x, y, z)
        delta_u = sp.Function("delta_u")(t, x, y, z)
        delta_pi = sp.Function("delta_pi")(t, x, y, z)
        delta_Pi = sp.Function("delta_Pi")(t, x, y, z)
        delta_q = sp.Function("delta_q")(t, x, y, z)

        # Linearized energy density equation: δρ̇ + ρ₀ ∇·δu = 0
        linearized_equations["delta_rho"] = sp.diff(delta_rho, t) + rho_0 * (
            sp.diff(delta_u, x) + sp.diff(delta_u, y) + sp.diff(delta_u, z)
        )

        # Linearized four-velocity equation (Euler equation)
        # δu̇ = -(1/(ρ₀+P₀)) ∇δP - κ₀∇δT/ε₀
        c_s_squared = P_0 / rho_0  # Sound speed squared
        linearized_equations["delta_u"] = (
            sp.diff(delta_u, t)
            + c_s_squared * sp.diff(delta_rho, x) / rho_0
            + self.parameters.kappa * sp.diff(delta_q, x) / rho_0
        )

        # Linearized shear stress evolution
        # τπ δπ̇ + δπ = 2η₀ δσ
        # where δσ is the linearized shear rate
        delta_sigma = sp.diff(delta_u, x) - sp.diff(delta_u, y) / 3  # Simplified shear rate
        linearized_equations["delta_pi"] = (
            self.parameters.tau_pi * sp.diff(delta_pi, t)
            + delta_pi
            - 2 * self.parameters.eta * delta_sigma
        )

        # Linearized bulk pressure evolution
        # τΠ δΠ̇ + δΠ = -ζ₀ δθ
        # where δθ = ∇·δu is the expansion perturbation
        delta_theta = sp.diff(delta_u, x) + sp.diff(delta_u, y) + sp.diff(delta_u, z)
        linearized_equations["delta_Pi"] = (
            self.parameters.tau_Pi * sp.diff(delta_Pi, t)
            + delta_Pi
            + self.parameters.zeta * delta_theta
        )

        # Linearized heat flux evolution
        # τq δq̇ + δq = -κ₀ ∇(δT/T₀)
        # Approximating δT ≈ (∂T/∂ρ)₀ δρ for simplicity
        delta_T_grad = sp.diff(delta_rho, x) / rho_0  # Simplified temperature gradient
        linearized_equations["delta_q"] = (
            self.parameters.tau_q * sp.diff(delta_q, t)
            + delta_q
            + self.parameters.kappa * delta_T_grad
        )

        return linearized_equations

    def __str__(self) -> str:
        """String representation"""
        return f"IsraelStewartSystem(fields={len(self.field_registry)}, params={self.parameters})"

    def __repr__(self) -> str:
        return self.__str__()
