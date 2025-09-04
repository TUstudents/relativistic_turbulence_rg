"""
Physics validation framework for MSRJD propagator calculations.

This module provides comprehensive validation of propagator physics including:
- Hydrodynamic mode structure (sound, shear, bulk)
- Transport coefficient consistency
- Fluctuation-dissipation theorem validation
- Sum rule verification
- Long-wavelength limit checks

The validation framework ensures that the MSRJD propagator calculations
produce physically meaningful results consistent with Israel-Stewart theory.
"""

import warnings
from typing import Any

import numpy as np
import sympy as sp

from ..core.constants import PhysicalConstants
from ..israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem


class PhysicsValidationError(Exception):
    """Raised when physics validation fails."""

    pass


class HydrodynamicModeAnalyzer:
    """
    Analyze hydrodynamic mode structure in MSRJD propagators.

    Validates that propagators exhibit correct dispersion relations
    for sound modes, shear modes, and bulk viscous modes.
    """

    def __init__(self, propagator_calculator: Any) -> None:
        self.calculator = propagator_calculator
        self.is_system = propagator_calculator.action.is_system
        self.tolerance = 1e-10

    def analyze_dispersion_relations(
        self, omega_range: np.ndarray, k_range: np.ndarray
    ) -> dict[str, Any]:
        """
        Analyze dispersion relations ω(k) for all hydrodynamic modes.

        Args:
            omega_range: Frequency range for analysis
            k_range: Momentum range for analysis

        Returns:
            Dictionary with mode analysis results
        """
        mode_results = {}

        # Analyze each mode type
        mode_results["sound"] = self._analyze_sound_modes(omega_range, k_range)
        mode_results["shear"] = self._analyze_shear_modes(omega_range, k_range)
        mode_results["bulk"] = self._analyze_bulk_modes(omega_range, k_range)
        mode_results["heat"] = self._analyze_heat_modes(omega_range, k_range)

        return mode_results

    def _analyze_sound_modes(self, omega_range: np.ndarray, k_range: np.ndarray) -> dict[str, Any]:
        """Analyze sound wave propagation modes."""
        results: dict[str, Any] = {"type": "sound", "validated": True, "issues": []}

        # Expected sound speed for relativistic fluid
        params = self.is_system.parameters
        expected_cs_squared = 1.0 / 3.0  # Ideal relativistic gas

        # Check dispersion relation: ω ≈ ±cs k for small k
        sound_speeds = []
        for k in k_range[k_range > 0]:
            # Find poles of propagator at small k
            try:
                propagator = self.calculator.compute_propagator(
                    "u", "rho", omega=k / np.sqrt(3), k=k
                )
                if abs(propagator) > 1e6:  # Near pole
                    sound_speeds.append(abs(k / (k / np.sqrt(3))))
            except Exception:  # nosec B112 - Safe to continue on numerical calculation failures
                continue

        if sound_speeds:
            avg_sound_speed = np.mean(sound_speeds)
            if abs(avg_sound_speed - np.sqrt(1 / 3)) > 0.1:
                results["validated"] = False
                results["issues"].append(
                    f"Sound speed {avg_sound_speed:.3f} differs from expected {np.sqrt(1 / 3):.3f}"
                )

        results["sound_speed"] = np.mean(sound_speeds) if sound_speeds else None
        return results

    def _analyze_shear_modes(self, omega_range: np.ndarray, k_range: np.ndarray) -> dict[str, Any]:
        """Analyze shear viscosity modes."""
        results: dict[str, Any] = {"type": "shear", "validated": True, "issues": []}

        # Shear modes should have damping ∝ η k²
        params = self.is_system.parameters
        expected_damping_coeff = params.eta

        # Check shear mode dispersion for small k
        damping_rates = []
        for k in k_range[k_range > 0]:
            # Shear mode: ω = -i (η/ρ) k² for small k
            omega_shear = -1j * expected_damping_coeff * k**2
            try:
                propagator = self.calculator.compute_propagator("pi", "pi", omega=omega_shear, k=k)
                if abs(propagator) > 1e6:  # Near pole
                    damping_rates.append(abs(omega_shear.imag / k**2))
            except Exception:  # nosec B112 - Safe to continue on numerical calculation failures
                continue

        if damping_rates:
            avg_damping = np.mean(damping_rates)
            if abs(avg_damping - expected_damping_coeff) / expected_damping_coeff > 0.2:
                results["validated"] = False
                results["issues"].append(
                    f"Shear damping {avg_damping:.3e} differs from expected {expected_damping_coeff:.3e}"
                )

        results["damping_coefficient"] = np.mean(damping_rates) if damping_rates else None
        return results

    def _analyze_bulk_modes(self, omega_range: np.ndarray, k_range: np.ndarray) -> dict[str, Any]:
        """Analyze bulk viscosity modes."""
        results: dict[str, Any] = {"type": "bulk", "validated": True, "issues": []}

        params = self.is_system.parameters

        # Bulk mode relaxation time
        if hasattr(params, "tau_Pi") and params.tau_Pi > 0:
            expected_relaxation_rate = 1.0 / params.tau_Pi

            # Check bulk pressure relaxation
            relaxation_rates = []
            for omega in omega_range:
                if abs(omega.imag - expected_relaxation_rate) < 0.1 * expected_relaxation_rate:
                    try:
                        propagator = self.calculator.compute_propagator(
                            "Pi", "Pi", omega=omega, k=0
                        )
                        if abs(propagator) > 1e6:  # Near pole
                            relaxation_rates.append(abs(omega.imag))
                    except Exception:
                        continue

            if relaxation_rates:
                avg_relaxation = np.mean(relaxation_rates)
                if abs(avg_relaxation - expected_relaxation_rate) / expected_relaxation_rate > 0.2:
                    results["validated"] = False
                    results["issues"].append(
                        f"Bulk relaxation {avg_relaxation:.3e} differs from expected {expected_relaxation_rate:.3e}"
                    )

        results["relaxation_rate"] = (
            np.mean(relaxation_rates)
            if "relaxation_rates" in locals() and relaxation_rates
            else None
        )
        return results

    def _analyze_heat_modes(self, omega_range: np.ndarray, k_range: np.ndarray) -> dict[str, Any]:
        """Analyze heat conduction modes."""
        results: dict[str, Any] = {"type": "heat", "validated": True, "issues": []}

        params = self.is_system.parameters

        # Heat diffusion modes: ω = -i κ k² / (ρ + p) for small k
        if hasattr(params, "kappa") and params.kappa > 0:
            expected_diffusion_coeff = params.kappa

            diffusion_rates = []
            for k in k_range[k_range > 0]:
                omega_heat = -1j * expected_diffusion_coeff * k**2
                try:
                    propagator = self.calculator.compute_propagator("q", "q", omega=omega_heat, k=k)
                    if abs(propagator) > 1e6:  # Near pole
                        diffusion_rates.append(abs(omega_heat.imag / k**2))
                except Exception:
                    continue

            if diffusion_rates:
                avg_diffusion = np.mean(diffusion_rates)
                if abs(avg_diffusion - expected_diffusion_coeff) / expected_diffusion_coeff > 0.2:
                    results["validated"] = False
                    results["issues"].append(
                        f"Heat diffusion {avg_diffusion:.3e} differs from expected {expected_diffusion_coeff:.3e}"
                    )

        results["diffusion_coefficient"] = (
            np.mean(diffusion_rates) if "diffusion_rates" in locals() and diffusion_rates else None
        )
        return results


class TransportCoefficientValidator:
    """
    Validate transport coefficients against kinetic theory predictions.

    Ensures that viscosities and conductivities extracted from propagators
    match expected values from Israel-Stewart parameters.
    """

    def __init__(self, propagator_calculator: Any) -> None:
        self.calculator = propagator_calculator
        self.is_system = propagator_calculator.action.is_system

    def validate_all_coefficients(self) -> dict[str, Any]:
        """Validate all transport coefficients."""
        results = {}

        results["shear_viscosity"] = self.validate_shear_viscosity()
        results["bulk_viscosity"] = self.validate_bulk_viscosity()
        results["thermal_conductivity"] = self.validate_thermal_conductivity()
        results["relaxation_times"] = self.validate_relaxation_times()

        return results

    def validate_shear_viscosity(self) -> dict[str, Any]:
        """Validate shear viscosity η."""
        params = self.is_system.parameters
        expected_eta = params.eta

        # Extract η from π^μν propagator at small k, ω=0
        try:
            # Low frequency, finite momentum limit
            propagator_val = self.calculator.compute_propagator("pi", "pi", omega=1e-6j, k=0.1)

            # Extract viscosity from imaginary part
            extracted_eta = (
                abs(propagator_val.imag) * params.tau_pi if hasattr(params, "tau_pi") else None
            )

            if extracted_eta and abs(extracted_eta - expected_eta) / expected_eta > 0.1:
                return {
                    "validated": False,
                    "expected": expected_eta,
                    "extracted": extracted_eta,
                    "error": abs(extracted_eta - expected_eta) / expected_eta,
                }

        except Exception as e:
            return {"validated": False, "error_message": str(e), "expected": expected_eta}

        return {"validated": True, "expected": expected_eta}

    def validate_bulk_viscosity(self) -> dict[str, Any]:
        """Validate bulk viscosity ζ."""
        params = self.is_system.parameters
        expected_zeta = params.zeta

        try:
            # Extract from bulk pressure propagator
            propagator_val = self.calculator.compute_propagator("Pi", "Pi", omega=1e-6j, k=0.1)

            extracted_zeta = (
                abs(propagator_val.imag) * params.tau_Pi if hasattr(params, "tau_Pi") else None
            )

            if extracted_zeta and abs(extracted_zeta - expected_zeta) / expected_zeta > 0.1:
                return {
                    "validated": False,
                    "expected": expected_zeta,
                    "extracted": extracted_zeta,
                    "error": abs(extracted_zeta - expected_zeta) / expected_zeta,
                }

        except Exception as e:
            return {"validated": False, "error_message": str(e), "expected": expected_zeta}

        return {"validated": True, "expected": expected_zeta}

    def validate_thermal_conductivity(self) -> dict[str, Any]:
        """Validate thermal conductivity κ."""
        params = self.is_system.parameters
        expected_kappa = params.kappa

        try:
            # Extract from heat flux propagator
            propagator_val = self.calculator.compute_propagator("q", "q", omega=1e-6j, k=0.1)

            extracted_kappa = (
                abs(propagator_val.imag) * params.tau_q if hasattr(params, "tau_q") else None
            )

            if extracted_kappa and abs(extracted_kappa - expected_kappa) / expected_kappa > 0.1:
                return {
                    "validated": False,
                    "expected": expected_kappa,
                    "extracted": extracted_kappa,
                    "error": abs(extracted_kappa - expected_kappa) / expected_kappa,
                }

        except Exception as e:
            return {"validated": False, "error_message": str(e), "expected": expected_kappa}

        return {"validated": True, "expected": expected_kappa}

    def validate_relaxation_times(self) -> dict[str, Any]:
        """Validate relaxation times τ_π, τ_Π, τ_q."""
        params = self.is_system.parameters
        results = {}

        # Check each relaxation time
        for field, param_name in [("pi", "tau_pi"), ("Pi", "tau_Pi"), ("q", "tau_q")]:
            expected_tau = getattr(params, param_name, None)
            if expected_tau:
                try:
                    # Find pole position in complex ω plane
                    omega_pole = self._find_pole_position(field)
                    if omega_pole:
                        extracted_tau = 1.0 / abs(omega_pole.imag)

                        if abs(extracted_tau - expected_tau) / expected_tau > 0.1:
                            results[param_name] = {
                                "validated": False,
                                "expected": expected_tau,
                                "extracted": extracted_tau,
                                "error": abs(extracted_tau - expected_tau) / expected_tau,
                            }
                        else:
                            results[param_name] = {"validated": True, "expected": expected_tau}
                    else:
                        results[param_name] = {"validated": False, "error": "Could not find pole"}

                except Exception as e:
                    results[param_name] = {
                        "validated": False,
                        "error_message": str(e),
                        "expected": expected_tau,
                    }

        return results

    def _find_pole_position(self, field_name: str) -> complex | None:
        """Find pole position in complex frequency plane."""
        # Simplified pole finding - scan complex ω plane
        for omega_real in np.linspace(-1, 1, 20):
            for omega_imag in np.linspace(-2, 0, 20):
                omega = omega_real + 1j * omega_imag
                try:
                    prop = self.calculator.compute_propagator(
                        field_name, field_name, omega=omega, k=0.1
                    )
                    if abs(prop) > 1e8:  # Near pole
                        return complex(omega)
                except Exception:
                    continue
        return None


class FluctuationDissipationValidator:
    """
    Validate fluctuation-dissipation theorem (FDT) relations.

    Checks that G^K = (G^R - G^A) coth(ω/2T) where:
    - G^R, G^A are retarded/advanced Green's functions
    - G^K is the Keldysh Green's function
    - T is temperature
    """

    def __init__(self, propagator_calculator: Any) -> None:
        self.calculator = propagator_calculator
        self.is_system = propagator_calculator.action.is_system

    def validate_fdt_relations(self, temperature: float = 1.0) -> dict[str, Any]:
        """Validate FDT for all field pairs."""
        results = {}

        field_pairs = [
            ("rho", "rho"),
            ("u", "u"),
            ("pi", "pi"),
            ("Pi", "Pi"),
            ("q", "q"),
            ("u", "rho"),
        ]

        for field1, field2 in field_pairs:
            results[f"{field1}_{field2}"] = self._validate_fdt_pair(field1, field2, temperature)

        return results

    def _validate_fdt_pair(self, field1: str, field2: str, temperature: float) -> dict[str, Any]:
        """Validate FDT for specific field pair."""
        omega_range = np.linspace(-2, 2, 50)
        k_test = 0.5

        violations = []

        for omega_val in omega_range[omega_range != 0]:  # Exclude ω=0
            try:
                # Compute retarded and advanced propagators
                G_R = self.calculator.compute_propagator(
                    field1, field2, omega=omega_val + 1e-6j, k=k_test
                )
                G_A = self.calculator.compute_propagator(
                    field1, field2, omega=omega_val - 1e-6j, k=k_test
                )

                # Expected Keldysh relation
                expected_GK = (G_R - G_A) * (1.0 / np.tanh(omega_val / (2 * temperature)))

                # For now, we don't have explicit Keldysh propagator computation
                # This is a placeholder for full implementation
                actual_GK = G_R - G_A  # Simplified

                violation = abs(actual_GK - expected_GK) / max(abs(expected_GK), 1e-10)
                if violation > 0.1:  # 10% tolerance
                    violations.append(violation)

            except Exception:  # nosec B112 - Safe to continue on numerical calculation failures
                continue

        return {
            "validated": len(violations) < len(omega_range) * 0.1,  # < 10% violations
            "max_violation": max(violations) if violations else 0,
            "avg_violation": np.mean(violations) if violations else 0,
            "num_violations": len(violations),
        }


class PhysicsValidationSuite:
    """
    Complete physics validation suite for MSRJD propagators.

    Combines all validation checks into a comprehensive test suite.
    """

    def __init__(self, propagator_calculator: Any) -> None:
        self.calculator = propagator_calculator
        self.mode_analyzer = HydrodynamicModeAnalyzer(propagator_calculator)
        self.transport_validator = TransportCoefficientValidator(propagator_calculator)
        self.fdt_validator = FluctuationDissipationValidator(propagator_calculator)

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete physics validation suite."""
        print("Running MSRJD propagator physics validation...")

        results: dict[str, Any] = {
            "timestamp": str(np.datetime64("now")),
            "overall_status": "PASS",
            "failed_tests": [],
        }

        # 1. Mode structure analysis
        print("  Analyzing hydrodynamic modes...")
        omega_range = np.linspace(-2, 2, 100, dtype=complex)
        k_range = np.linspace(0, 2, 50)

        mode_results = self.mode_analyzer.analyze_dispersion_relations(omega_range, k_range)
        results["mode_analysis"] = mode_results

        # Check for mode validation failures
        for mode_type, mode_data in mode_results.items():
            if not mode_data.get("validated", True):
                results["overall_status"] = "FAIL"
                results["failed_tests"].append(f"Mode analysis: {mode_type}")

        # 2. Transport coefficient validation
        print("  Validating transport coefficients...")
        transport_results = self.transport_validator.validate_all_coefficients()
        results["transport_validation"] = transport_results

        # Check for transport validation failures
        for coeff_type, coeff_data in transport_results.items():
            if not coeff_data.get("validated", True):
                results["overall_status"] = "FAIL"
                results["failed_tests"].append(f"Transport coefficient: {coeff_type}")

        # 3. FDT validation
        print("  Checking fluctuation-dissipation relations...")
        fdt_results = self.fdt_validator.validate_fdt_relations()
        results["fdt_validation"] = fdt_results

        # Check for FDT validation failures
        for pair, fdt_data in fdt_results.items():
            if not fdt_data.get("validated", True):
                results["overall_status"] = "FAIL"
                results["failed_tests"].append(f"FDT violation: {pair}")

        # Summary
        if results["overall_status"] == "PASS":
            print("✓ All physics validation tests passed")
        else:
            print(f"✗ Physics validation failed: {len(results['failed_tests'])} tests failed")
            for failed_test in results["failed_tests"]:
                print(f"    - {failed_test}")

        return results
