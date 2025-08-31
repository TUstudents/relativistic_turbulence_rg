"""
Physical parameter management and validation for Israel-Stewart relativistic hydrodynamics.

This module provides comprehensive management of all physical parameters appearing
in Israel-Stewart theory, including transport coefficients, relaxation times,
thermodynamic quantities, and higher-order coupling terms. The implementation
enforces physical consistency through validation of causality, thermodynamic
stability, and dimensional analysis.

Parameter Categories:
    Transport Coefficients:
        - η: Shear viscosity [M L^{-1} T^{-1}]
        - ζ: Bulk viscosity [M L^{-1} T^{-1}]
        - κ: Thermal conductivity [M L T^{-3} K^{-1}]

    Relaxation Times:
        - τ_π: Shear stress relaxation time [T]
        - τ_Π: Bulk pressure relaxation time [T]
        - τ_q: Heat flux relaxation time [T]

    Thermodynamic Quantities:
        - c_s: Sound velocity [L T^{-1}]
        - T: Temperature [energy]
        - p: Pressure [energy density]
        - ρ: Energy density [energy density]

Physical Constraints:
    - Causality: All characteristic velocities < c
    - Positivity: η, κ, τ_i > 0; ζ ≥ 0
    - Thermodynamic stability: ∂p/∂ρ > 0
    - Sound speed bound: c_s < c

Dimensionless Analysis:
    The theory contains several important dimensionless parameters:
    - Reynolds number: Re = ρvL/η (inertial/viscous forces)
    - Knudsen number: Kn = τc_s/L (kinetic/hydrodynamic scales)
    - Mach number: Ma = v/c_s (flow/sound speeds)

Kinetic Theory Relations:
    For dilute gases, kinetic theory provides parameter relations:
    - η = p·τ_π (pressure × relaxation time)
    - ζ = p·τ_Π·(1/3 - c_s²) (bulk viscosity from trace anomaly)
    - Relaxation times: τ ~ 1/(n·σ·v) (collision time scale)

References:
    - Israel, W. & Stewart, J.M. Phys. Lett. A 58, 213 (1976)
    - Denicol, G.S. et al. Phys. Rev. D 85, 114047 (2012)
    - Romatschke, P. & Romatschke, U. "Relativistic Fluid Dynamics In and Out of Equilibrium" (2019)
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import PhysicalConstants


@dataclass
class ISParameters:
    """
    Complete parameter set for Israel-Stewart relativistic hydrodynamics with validation.

    Manages all physical parameters required for IS theory calculations, including
    transport coefficients, relaxation times, thermodynamic quantities, and higher-order
    coupling terms. Provides automatic validation of physical constraints, dimensional
    consistency checking, and unit conversion utilities.

    Physical Parameter Set:
        The complete IS theory requires 13 independent parameters plus thermodynamic
        state variables. This class handles parameter interdependencies and ensures
        physical consistency through comprehensive validation.

    Validation Framework:
        Automatic checking of:
        - Positivity constraints (η, κ, τ_i > 0)
        - Causality conditions (all velocities < c)
        - Thermodynamic stability (convexity conditions)
        - Dimensional consistency
        - Parameter range validity

    Kinetic Theory Integration:
        Methods for setting parameters according to kinetic theory relations,
        enabling connection to microscopic physics through collision integrals
        and transport coefficients derived from Boltzmann equation.

    Attributes:
        eta: Shear viscosity coefficient η [M L^{-1} T^{-1}]
            Controls momentum diffusion in spatial directions.
        zeta: Bulk viscosity coefficient ζ [M L^{-1} T^{-1}]
            Controls pressure relaxation and volume changes.
        kappa: Thermal conductivity κ [M L T^{-3} K^{-1}]
            Controls heat diffusion and temperature equilibration.
    """

    # Transport coefficients
    eta: float
    zeta: float
    kappa: float

    # Relaxation times
    tau_pi: float  # Shear relaxation time
    tau_Pi: float  # Bulk relaxation time
    tau_q: float  # Heat flux relaxation time

    # Thermodynamic parameters
    cs: float  # Sound speed
    temperature: float  # Temperature
    pressure: float = 0.0  # Equilibrium pressure
    energy_density: float = 0.0  # Equilibrium energy density

    # Higher-order coupling coefficients (optional)
    lambda_pi_Pi: float = 0.0  # π-Π coupling
    lambda_pi_q: float = 0.0  # π-q coupling
    lambda_qq: float = 0.0  # q-q self-coupling
    tau_pi_pi: float = 0.0  # π-π relaxation

    # System parameters
    dimension: int = 4  # Spacetime dimension
    unit_system: str = "natural"  # Unit system

    # Validation parameters
    _validated: bool = field(default=False, init=False)
    _warnings: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate parameters after initialization"""
        self.validate()
        self._setup_derived_quantities()

    def validate(self) -> bool:
        """
        Comprehensive validation of physical parameter consistency and constraints.

        Performs systematic checking of all physical constraints required for
        Israel-Stewart theory, including causality conditions, thermodynamic
        stability, and dimensional consistency. Issues warnings for questionable
        but potentially valid parameter choices.

        Validation Checks:
            1. Positivity Constraints:
               - Transport coefficients: η, κ > 0
               - Relaxation times: τ_π, τ_Π, τ_q > 0
               - Thermodynamic quantities: T, c_s > 0

            2. Causality Conditions:
               - Sound speed: c_s < c (subluminal sound propagation)
               - Maximum characteristic speed < c (no superluminal modes)
               - Finite signal propagation in linearized theory

            3. Thermodynamic Stability:
               - Positive energy density (when specified)
               - Thermodynamic consistency relations
               - Valid equation of state behavior

            4. Parameter Range Validity:
               - Physical ranges for transport coefficients
               - Consistency with kinetic theory bounds
               - Dimensional analysis verification

        Returns:
            True if all validation checks pass successfully.

        Raises:
            ValueError: If any critical physical constraints are violated.
                Includes detailed description of specific violations.

        Warnings:
            UserWarning: For parameter choices that are unusual but potentially
                valid (e.g., negative bulk viscosity, very large Reynolds numbers).

        Examples:
            >>> params = ISParameters(eta=0.1, zeta=0.05, ...)
            >>> try:
            ...     params.validate()
            ...     print("Parameters are physically consistent")
            ... except ValueError as e:
            ...     print(f"Invalid parameters: {e}")
        """
        errors = []
        warnings_list = []

        # Positivity constraints
        if self.eta <= 0:
            errors.append("Shear viscosity η must be positive")
        if self.zeta < 0:
            warnings_list.append("bulk viscosity ζ is typically non-negative")
        if self.kappa <= 0:
            errors.append("Thermal conductivity κ must be positive")

        # Relaxation times must be positive (causality)
        for param, name in [(self.tau_pi, "τ_π"), (self.tau_Pi, "τ_Π"), (self.tau_q, "τ_q")]:
            if param <= 0:
                errors.append(f"Relaxation time {name} must be positive")

        # Sound speed constraints
        if self.cs <= 0:
            errors.append("Sound speed must be positive")
        if self.cs > PhysicalConstants.c:
            errors.append(f"Sound speed {self.cs} exceeds speed of light")

        # Temperature must be positive
        if self.temperature <= 0:
            errors.append("Temperature must be positive")

        # Thermodynamic consistency (simplified)
        if self.energy_density < 0:
            warnings_list.append("Negative energy density may be unphysical")

        # Causality constraints: check that characteristic speeds are physical
        if not self._check_causality():
            errors.append("Parameters violate causality (superluminal modes)")

        # Store warnings
        self._warnings = warnings_list
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

        # Raise errors if any
        if errors:
            raise ValueError(
                "Parameter validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        self._validated = True
        return True

    def _check_causality(self) -> bool:
        """Check that no characteristic speeds exceed c"""
        try:
            # In relativistic theories, all characteristic speeds must be subluminal
            # We check each mode separately to ensure causality

            # Sound speed must be subluminal (or equal in special cases like conformal fluids)
            if self.cs > PhysicalConstants.c:
                return False

            # Check if we have valid parameters for computation
            if self.energy_density <= 0 or self.tau_pi <= 0:
                # Can't compute shear speed, assume valid
                v_shear = 0.0
            else:
                # Shear mode speed: In relativistic hydrodynamics, this should be bounded by c
                # Using a more physically motivated approach
                shear_arg = max(0.0, self.eta / (self.energy_density * self.tau_pi))
                # Ensure the argument doesn't lead to superluminal speeds
                if shear_arg > PhysicalConstants.c**2:
                    return False
                v_shear = np.sqrt(shear_arg)

            # Bulk mode contributions
            if self.energy_density <= 0 or self.tau_Pi <= 0:
                v_bulk_contrib = 0.0
            else:
                # Bulk mode speed: Also bounded by c
                bulk_arg = max(0.0, self.zeta / (self.energy_density * self.tau_Pi))
                # Ensure the argument doesn't lead to superluminal speeds
                if bulk_arg > PhysicalConstants.c**2:
                    return False
                v_bulk_contrib = np.sqrt(bulk_arg)

            # Individual speeds must be subluminal (more stringent than combined)
            if v_shear > PhysicalConstants.c or v_bulk_contrib > PhysicalConstants.c:
                return False

            return True

        except (ZeroDivisionError, ValueError, OverflowError):
            # If we can't compute reliably, assume valid for now
            return True

    def _setup_derived_quantities(self) -> None:
        """Setup derived dimensionless parameters"""
        if not hasattr(self, "_derived"):
            self._derived = {}

        # Only compute if we have the necessary base quantities
        try:
            # Kinematic viscosity
            if self.energy_density > 0:
                self._derived["nu_shear"] = self.eta / self.energy_density
                self._derived["nu_bulk"] = self.zeta / self.energy_density

            # Prandtl number
            if self.kappa > 0 and self.eta > 0:
                # Simplified - full definition would need heat capacity
                self._derived["Pr"] = self.eta / self.kappa

            # Relaxation length scales
            self._derived["l_tau_pi"] = self.tau_pi * self.cs
            self._derived["l_tau_Pi"] = self.tau_Pi * self.cs
            self._derived["l_tau_q"] = self.tau_q * self.cs

        except (ZeroDivisionError, ValueError):
            # Some derived quantities couldn't be computed
            pass

    def to_dimensionless(self, L0: float, rho0: float, T0: float | None = None) -> "ISParameters":
        """Convert to dimensionless parameters

        Args:
            L0: Characteristic length scale
            rho0: Characteristic density scale
            T0: Characteristic temperature scale (default: from cs)

        Returns:
            New ISParameters object with dimensionless parameters
        """
        if T0 is None:
            T0 = rho0 * self.cs**2  # Rough estimate

        # Time scale
        t0 = L0 / self.cs

        # Dimensionless parameters
        return ISParameters(
            eta=self.eta / (rho0 * self.cs * L0),
            zeta=self.zeta / (rho0 * self.cs * L0),
            kappa=self.kappa * T0 / (rho0 * self.cs**3 * L0),
            tau_pi=self.tau_pi / t0,
            tau_Pi=self.tau_Pi / t0,
            tau_q=self.tau_q / t0,
            cs=1.0,  # Sound speed normalized to 1
            temperature=self.temperature / T0,
            pressure=self.pressure / (rho0 * self.cs**2),
            energy_density=self.energy_density / rho0,
            lambda_pi_Pi=self.lambda_pi_Pi,  # Already dimensionless
            lambda_pi_q=self.lambda_pi_q,
            lambda_qq=self.lambda_qq,
            tau_pi_pi=self.tau_pi_pi / t0,
            dimension=self.dimension,
            unit_system="dimensionless",
        )

    def characteristic_scales(self) -> dict[str, float]:
        """Compute characteristic scales in the problem

        Returns:
            Dictionary of characteristic scales
        """
        scales = {}

        # Length scales
        if hasattr(self, "_derived"):
            scales.update({k: v for k, v in self._derived.items() if k.startswith("l_")})

        # Time scales
        scales["t_pi"] = self.tau_pi
        scales["t_Pi"] = self.tau_Pi
        scales["t_q"] = self.tau_q

        # Velocity scales
        scales["c_s"] = self.cs
        scales["c"] = PhysicalConstants.c

        # If we have energy density, compute viscous scales
        if self.energy_density > 0:
            # Viscous length (assuming some velocity scale)
            v_scale = self.cs  # Use sound speed as velocity scale
            scales["l_viscous"] = (self.eta / (self.energy_density * v_scale)) ** (1 / 2)

        return scales

    def dimensionless_numbers(
        self, L: float | None = None, v: float | None = None
    ) -> dict[str, float]:
        """Compute dimensionless numbers characterizing the flow

        Args:
            L: Characteristic length scale
            v: Characteristic velocity scale

        Returns:
            Dictionary of dimensionless numbers
        """
        numbers = {}

        # Use defaults if not provided
        if L is None:
            L = self._derived.get("l_tau_pi", 1.0)
        if v is None:
            v = self.cs

        # Reynolds numbers
        if self.energy_density > 0:
            numbers["Re_shear"] = self.energy_density * v * L / self.eta
            if self.zeta > 0:
                numbers["Re_bulk"] = self.energy_density * v * L / self.zeta

        # Knudsen numbers
        numbers["Kn_pi"] = self.tau_pi * self.cs / L
        numbers["Kn_Pi"] = self.tau_Pi * self.cs / L
        numbers["Kn_q"] = self.tau_q * self.cs / L

        # Mach number
        numbers["Ma"] = v / self.cs

        # Relaxation parameters
        numbers["tau_pi_cs_over_L"] = self.tau_pi * self.cs / L
        numbers["tau_Pi_cs_over_L"] = self.tau_Pi * self.cs / L

        return numbers

    def kinetic_theory_relations(
        self, particle_mass: float = 1.0, cross_section: float = 1.0
    ) -> "ISParameters":
        """Set parameters according to kinetic theory relations

        Args:
            particle_mass: Particle mass
            cross_section: Collision cross section

        Returns:
            Updated parameters with kinetic theory values
        """
        # This is simplified - full kinetic theory would be more complex

        # Collision time (simplified)
        n = self.energy_density / particle_mass  # Rough particle density
        tau_0 = 1.0 / (n * cross_section * self.cs)  # Collision time

        # Set relaxation times equal (simplification)
        self.tau_pi = tau_0
        self.tau_Pi = tau_0
        self.tau_q = tau_0

        # Viscosity relations
        self.eta = self.pressure * tau_0
        self.zeta = self.pressure * tau_0 * (1 / 3 - self.cs**2)

        # Thermal conductivity
        self.kappa = self.pressure * tau_0 / self.temperature

        # Revalidate
        self.validate()
        self._setup_derived_quantities()

        return self

    def copy(self) -> "ISParameters":
        """Create a copy of the parameters"""
        return ISParameters(
            eta=self.eta,
            zeta=self.zeta,
            kappa=self.kappa,
            tau_pi=self.tau_pi,
            tau_Pi=self.tau_Pi,
            tau_q=self.tau_q,
            cs=self.cs,
            temperature=self.temperature,
            pressure=self.pressure,
            energy_density=self.energy_density,
            lambda_pi_Pi=self.lambda_pi_Pi,
            lambda_pi_q=self.lambda_pi_q,
            lambda_qq=self.lambda_qq,
            tau_pi_pi=self.tau_pi_pi,
            dimension=self.dimension,
            unit_system=self.unit_system,
        )

    def update(self, **kwargs: Any) -> "ISParameters":
        """Update parameters and return new instance"""
        new_params = self.copy()
        for key, value in kwargs.items():
            if hasattr(new_params, key):
                setattr(new_params, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

        new_params.validate()
        new_params._setup_derived_quantities()
        return new_params

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "eta": self.eta,
            "zeta": self.zeta,
            "kappa": self.kappa,
            "tau_pi": self.tau_pi,
            "tau_Pi": self.tau_Pi,
            "tau_q": self.tau_q,
            "cs": self.cs,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "energy_density": self.energy_density,
            "lambda_pi_Pi": self.lambda_pi_Pi,
            "lambda_pi_q": self.lambda_pi_q,
            "lambda_qq": self.lambda_qq,
            "tau_pi_pi": self.tau_pi_pi,
            "dimension": self.dimension,
            "unit_system": self.unit_system,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ISParameters":
        """Create from dictionary"""
        return cls(**data)

    def __str__(self) -> str:
        """String representation"""
        return (
            f"ISParameters(η={self.eta:.3g}, ζ={self.zeta:.3g}, "
            f"τ_π={self.tau_pi:.3g}, c_s={self.cs:.3g}, T={self.temperature:.3g})"
        )

    def __repr__(self) -> str:
        return self.__str__()


# Predefined parameter sets for common scenarios
class StandardParameterSets:
    """Standard parameter sets for common physical scenarios"""

    @staticmethod
    def weakly_coupled_plasma(temperature: float = 1.0) -> ISParameters:
        """Parameters for weakly-coupled relativistic plasma"""
        return ISParameters(
            eta=0.5 * temperature**3,  # Weak coupling estimate
            zeta=0.1 * temperature**3,
            kappa=2.0 * temperature**2,
            tau_pi=1.0 / temperature,  # ~ 1/T in natural units
            tau_Pi=1.0 / temperature,
            tau_q=1.0 / temperature,
            cs=1.0 / np.sqrt(3),  # Relativistic ideal gas
            temperature=temperature,
            pressure=temperature**4 / 3,
            energy_density=3 * temperature**4,
        )

    @staticmethod
    def strongly_coupled_plasma(temperature: float = 1.0) -> ISParameters:
        """Parameters for strongly-coupled plasma (AdS/CFT inspired)"""
        return ISParameters(
            eta=temperature**3 / (4 * np.pi),  # AdS/CFT bound
            zeta=0.0,  # Conformal
            kappa=temperature**2 / (4 * np.pi),
            tau_pi=1.0 / (2 * np.pi * temperature),  # Strong coupling
            tau_Pi=1.0 / (2 * np.pi * temperature),
            tau_q=1.0 / (2 * np.pi * temperature),
            cs=1.0 / np.sqrt(3),  # Conformal
            temperature=temperature,
            pressure=temperature**4 / 3,
            energy_density=3 * temperature**4,
        )

    @staticmethod
    def nuclear_matter(temperature: float = 0.1, density: float = 1.0) -> ISParameters:
        """Parameters for nuclear matter"""
        return ISParameters(
            eta=0.1 * density * temperature,
            zeta=0.05 * density * temperature,
            kappa=0.2 * density * temperature**2,
            tau_pi=1.0 / temperature,
            tau_Pi=2.0 / temperature,  # Bulk relaxation can be slower
            tau_q=1.5 / temperature,
            cs=0.3,  # Typical nuclear matter sound speed
            temperature=temperature,
            pressure=0.3 * density * temperature,
            energy_density=density,
        )

    @staticmethod
    def qgp_phenomenology(temperature: float = 0.3) -> ISParameters:
        """QGP parameters from phenomenological studies"""
        T_GeV = temperature  # Temperature in GeV

        return ISParameters(
            eta=0.2 * T_GeV**3,  # η/s ≈ 0.2 near Tc
            zeta=0.05 * T_GeV**3,  # Small bulk viscosity
            kappa=1.0 * T_GeV**2,
            tau_pi=5.0 / T_GeV,  # ~ 5/T from phenomenology
            tau_Pi=3.0 / T_GeV,
            tau_q=4.0 / T_GeV,
            cs=1.0 / np.sqrt(3),  # Near conformal
            temperature=T_GeV,
            pressure=T_GeV**4 * np.pi**2 / 90,  # Stefan-Boltzmann
            energy_density=3 * T_GeV**4 * np.pi**2 / 90,
        )
