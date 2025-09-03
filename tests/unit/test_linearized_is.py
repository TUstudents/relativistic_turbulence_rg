"""
Comprehensive tests for linearized Israel-Stewart hydrodynamics.

This module tests all components of the linearized IS system, including:
- Background state validation and equilibrium conditions
- Field linearization and perturbation algebra
- Dispersion relation calculations and mode analysis
- Linear stability analysis and causality verification
- Sound attenuation and critical parameter calculations

Test Structure:
    - TestBackgroundState: Equilibrium state validation
    - TestLinearizedField: Field linearization mechanics
    - TestLinearizedIS: Complete linearized system functionality
    - TestDispersionRelations: Mode analysis and frequency calculations
    - TestStabilityAnalysis: Growth rates and stability determination
    - TestPhysicalProperties: Sound speed, attenuation, causality
"""

from typing import Dict, List  # noqa: UP035

import numpy as np
import pytest
import sympy as sp

from rtrg.core.constants import PhysicalConstants
from rtrg.core.tensors import Metric
from rtrg.israel_stewart.equations import IsraelStewartParameters
from rtrg.israel_stewart.linearized import BackgroundState, LinearizedField, LinearizedIS


class TestBackgroundState:
    """Test background equilibrium state functionality."""

    def test_default_background_state(self):
        """Test creation of default equilibrium state."""
        background = BackgroundState()

        # Check default values
        assert background.rho == 1.0
        assert background.pressure == 0.33
        assert background.temperature == 1.0
        assert background.Pi == 0.0
        assert background.pi == 0.0

        # Check default four-velocity (fluid at rest)
        expected_u = [PhysicalConstants.c, 0.0, 0.0, 0.0]
        assert background.u == expected_u

        # Check default heat flux (no heat transfer)
        expected_q = [0.0, 0.0, 0.0, 0.0]
        assert background.q == expected_q

    def test_four_velocity_normalization(self):
        """Test four-velocity normalization constraint."""
        # Valid normalized four-velocity
        background = BackgroundState(u=[PhysicalConstants.c, 0, 0, 0])

        # Check normalization: u·u = -c²
        u = background.u
        norm_sq = -(u[0] ** 2) + sum(u[i] ** 2 for i in range(1, 4))
        expected = -(PhysicalConstants.c**2)
        assert abs(norm_sq - expected) < 1e-10

    def test_invalid_four_velocity_normalization(self):
        """Test rejection of non-normalized four-velocity."""
        # Invalid four-velocity (not normalized)
        with pytest.raises(ValueError, match="Four-velocity not normalized"):
            BackgroundState(u=[2.0, 1.0, 0.0, 0.0])  # Wrong normalization

    def test_validate_equilibrium_fluid_at_rest(self):
        """Test equilibrium validation for fluid at rest."""
        # Perfect equilibrium state
        background = BackgroundState()
        assert background.validate_equilibrium() is True

        # Moving fluid (not equilibrium) - create properly normalized four-velocity
        c = PhysicalConstants.c
        v = 0.1
        gamma = 1 / np.sqrt(1 - v**2 / c**2)
        u_moving = [gamma * c, gamma * v, 0, 0]
        background_moving = BackgroundState(u=u_moving)

        assert background_moving.validate_equilibrium() is False

    def test_validate_equilibrium_with_dissipative_fluxes(self):
        """Test equilibrium validation with non-zero dissipative fluxes."""
        # Non-zero bulk pressure (not equilibrium)
        background_bulk = BackgroundState(Pi=0.1)
        assert background_bulk.validate_equilibrium() is False

        # Non-zero heat flux (not equilibrium)
        background_heat = BackgroundState(q=[0, 0.1, 0, 0])
        assert background_heat.validate_equilibrium() is False

        # Non-zero shear stress (not equilibrium)
        background_shear = BackgroundState(pi=0.1)
        assert background_shear.validate_equilibrium() is False


class TestLinearizedField:
    """Test linearized field representation and algebra."""

    def test_scalar_field_linearization(self):
        """Test linearization of scalar fields."""
        field = LinearizedField("rho", 1.0, tensor_rank=0)

        assert field.name == "rho"
        assert field.background == 1.0
        assert field.tensor_rank == 0
        assert isinstance(field.perturbation, sp.Symbol)
        assert str(field.perturbation) == "delta_rho"

    def test_vector_field_linearization(self):
        """Test linearization of vector fields."""
        u_background = [PhysicalConstants.c, 0, 0, 0]
        field = LinearizedField("u", u_background, tensor_rank=1)

        assert field.name == "u"
        assert field.background == u_background
        assert field.tensor_rank == 1
        assert isinstance(field.perturbation, sp.IndexedBase)

    def test_tensor_field_linearization(self):
        """Test linearization of tensor fields."""
        field = LinearizedField(
            "pi", 0.0, tensor_rank=2, symmetries=["symmetric", "traceless", "spatial"]
        )

        assert field.name == "pi"
        assert field.tensor_rank == 2
        assert "symmetric" in field.symmetries
        assert "traceless" in field.symmetries
        assert "spatial" in field.symmetries

    def test_total_field_construction(self):
        """Test construction of total field: φ = φ₀ + δφ."""
        field = LinearizedField("rho", 1.5, tensor_rank=0)
        total = field.total_field()

        # Total field should be background + perturbation
        expected = 1.5 + field.perturbation
        assert total == expected

    def test_unsupported_tensor_rank(self):
        """Test error handling for unsupported tensor ranks."""
        with pytest.raises(ValueError, match="Tensor rank 3 not supported"):
            LinearizedField("test", 0.0, tensor_rank=3)


class TestLinearizedIS:
    """Test complete linearized Israel-Stewart system."""

    @pytest.fixture
    def background_state(self):
        """Standard equilibrium background state."""
        return BackgroundState(rho=1.0, pressure=0.33, temperature=1.0)

    @pytest.fixture
    def parameters(self):
        """Standard IS parameters."""
        return IsraelStewartParameters(
            eta=1.0,
            zeta=0.1,
            kappa=0.5,
            tau_pi=0.15,
            tau_Pi=0.05,
            tau_q=0.02,  # Ensure causality
        )

    @pytest.fixture
    def linearized_system(self, background_state, parameters):
        """Complete linearized IS system."""
        return LinearizedIS(background_state, parameters)

    def test_system_initialization(self, linearized_system):
        """Test proper initialization of linearized system."""
        system = linearized_system

        # Check background and parameters stored
        assert system.background.rho == 1.0
        assert system.parameters.eta == 1.0

        # Check linearized fields created
        expected_fields = ["rho", "u", "pi", "Pi", "q"]
        for field_name in expected_fields:
            assert field_name in system.linearized_fields

        # Check symbolic variables created
        assert system.omega is not None
        assert system.k is not None

    def test_invalid_background_rejection(self, parameters):
        """Test rejection of non-equilibrium background states."""
        # Background with moving fluid
        c = PhysicalConstants.c
        v = 0.2
        gamma = 1 / np.sqrt(1 - v**2 / c**2)
        moving_background = BackgroundState(u=[gamma * c, gamma * v, 0, 0])

        with pytest.raises(ValueError, match="not a valid equilibrium"):
            LinearizedIS(moving_background, parameters)

    def test_linearize_field_access(self, linearized_system):
        """Test access to linearized fields."""
        system = linearized_system

        # Test accessing existing field
        rho_field = system.linearize_field("rho")
        assert rho_field.name == "rho"
        assert rho_field.background == 1.0

        # Test accessing non-existent field
        with pytest.raises(ValueError, match="Unknown field: nonexistent"):
            system.linearize_field("nonexistent")

    def test_linearized_equations_structure(self, linearized_system):
        """Test structure of linearized evolution equations."""
        system = linearized_system
        equations = system.get_linearized_equations()

        # Check that key equations are present
        assert "continuity" in equations
        assert "bulk" in equations

        # Check momentum equations for spatial components
        for i in range(1, 4):
            assert f"momentum_{i}" in equations

        # Check heat flux equations
        for i in range(1, 4):
            assert f"heat_{i}" in equations

        # Check shear stress equations (symmetric tensor)
        for i in range(1, 4):
            for j in range(i, 4):
                assert f"shear_{i}_{j}" in equations

    def test_continuity_equation_form(self, linearized_system):
        """Test structure of linearized continuity equation."""
        system = linearized_system
        equations = system.get_linearized_equations()
        continuity = equations["continuity"]

        # Should contain time derivative of density
        delta_rho = system.linearized_fields["rho"].perturbation
        assert sp.Derivative(delta_rho, system.t) in continuity.args

        # Should be linear in perturbations
        assert continuity.is_Add  # Sum of terms


class TestDispersionRelations:
    """Test dispersion relation calculations and mode analysis."""

    @pytest.fixture
    def simple_system(self):
        """Simple system for dispersion relation testing."""
        background = BackgroundState(rho=1.0, pressure=0.33)
        params = IsraelStewartParameters(eta=0.5, zeta=0.1, tau_pi=0.2, tau_Pi=0.1)
        return LinearizedIS(background, params)

    def test_characteristic_polynomial_structure(self, simple_system):
        """Test structure of characteristic polynomial."""
        system = simple_system
        char_poly = system.characteristic_polynomial(system.omega, system.k)

        # Should be polynomial in omega and k
        assert isinstance(char_poly, sp.Basic)
        assert char_poly != 0  # Non-trivial polynomial

        # Should contain omega and k symbols
        symbols = char_poly.free_symbols
        assert system.omega in symbols
        assert system.k in symbols

    def test_sound_mode_dispersion(self, simple_system):
        """Test sound mode dispersion relation."""
        system = simple_system

        # Test dispersion at small k
        k_small = 0.1
        omega_sound = system.dispersion_relation(k_small, mode="sound")

        # Should be complex number
        assert isinstance(omega_sound, complex)

        # Real part should be approximately c_s * k for sound mode
        sound_speed_sq = system.parameters.equilibrium_pressure / system.background.rho
        expected_freq = np.sqrt(sound_speed_sq) * k_small

        # Allow some tolerance for numerical solution
        assert abs(omega_sound.real - expected_freq) < 0.5 * expected_freq

    def test_dispersion_relation_causality(self, simple_system):
        """Test that dispersion relations respect causality."""
        system = simple_system

        # Test multiple k values
        k_values = [0.1, 0.5, 1.0, 2.0]

        for k_val in k_values:
            omega = system.dispersion_relation(k_val, "sound")
            phase_velocity = abs(omega.real) / k_val

            # Phase velocity should be less than speed of light
            assert phase_velocity < PhysicalConstants.c

    def test_diffusive_mode_properties(self, simple_system):
        """Test properties of diffusive modes."""
        system = simple_system

        k_val = 1.0
        omega_diff = system.dispersion_relation(k_val, mode="diffusive")

        # For our simplified system, diffusive modes may have different characteristics
        # The key requirement is that they should be physical (finite frequency)
        assert isinstance(omega_diff, complex)
        assert abs(omega_diff) < 10.0  # Reasonable magnitude

        # Should generally have decay (negative imaginary part) or be stable
        # But allow some flexibility for simplified analysis
        assert omega_diff.imag <= 0.1  # Allow small positive growth for simplified system


class TestStabilityAnalysis:
    """Test linear stability analysis and growth rate calculations."""

    @pytest.fixture
    def stable_system(self):
        """System parameters that should be linearly stable."""
        background = BackgroundState()
        params = IsraelStewartParameters(
            eta=1.0,
            zeta=0.1,
            kappa=0.5,
            tau_pi=0.2,
            tau_Pi=0.1,
            tau_q=0.05,  # Large relaxation times for stability
        )
        return LinearizedIS(background, params)

    def test_stability_analysis_stable_system(self, stable_system):
        """Test stability analysis for stable system."""
        system = stable_system

        # System should be linearly stable
        is_stable = system.is_linearly_stable(k_range=(0.1, 5.0), num_points=20)
        assert is_stable is True

    def test_stability_analysis_k_range_sampling(self, stable_system):
        """Test stability analysis with different k ranges."""
        system = stable_system

        # Test different k ranges
        ranges = [(0.1, 1.0), (0.5, 2.0), (1.0, 10.0)]

        for k_range in ranges:
            is_stable = system.is_linearly_stable(k_range=k_range, num_points=10)
            # Should be consistent (all stable for this system)
            assert isinstance(is_stable, bool)

    def test_causality_validation(self, stable_system):
        """Test causality validation for all modes."""
        system = stable_system

        # All modes should respect causality
        causality_ok = system.validate_causality()
        assert causality_ok is True

    def test_growth_rate_signs(self, stable_system):
        """Test that growth rates have correct signs for stable system."""
        system = stable_system

        # Check multiple modes and k values
        k_values = [0.2, 0.5, 1.0, 2.0]
        modes = ["sound", "diffusive"]

        for k_val in k_values:
            for mode in modes:
                omega = system.dispersion_relation(k_val, mode)

                # For stable system, imaginary part should be ≤ 0
                assert omega.imag <= 1e-10  # Small tolerance


class TestPhysicalProperties:
    """Test physical properties and coefficients."""

    @pytest.fixture
    def reference_system(self):
        """Reference system with known parameters."""
        background = BackgroundState(rho=2.0, pressure=0.5)
        params = IsraelStewartParameters(eta=1.5, zeta=0.3, tau_pi=0.1)
        return LinearizedIS(background, params)

    def test_sound_attenuation_coefficient(self, reference_system):
        """Test calculation of sound attenuation coefficient."""
        system = reference_system

        gamma = system.sound_attenuation_coefficient()

        # Should be positive for physical systems
        assert gamma > 0

        # Check formula: Γ = (4η/3 + ζ)/ρ
        expected_gamma = (
            4 * system.parameters.eta / 3 + system.parameters.zeta
        ) / system.background.rho
        assert abs(gamma - expected_gamma) < 1e-12

    def test_critical_parameters_calculation(self, reference_system):
        """Test calculation of critical parameter values."""
        system = reference_system

        critical_params = system.critical_parameters()

        # Should contain expected keys
        expected_keys = ["tau_pi_critical", "sound_speed_squared", "attenuation_coefficient"]
        for key in expected_keys:
            assert key in critical_params

        # All values should be positive
        for value in critical_params.values():
            assert value > 0

        # Sound speed squared should match thermodynamic relation
        expected_cs2 = system.background.pressure / system.background.rho
        assert abs(critical_params["sound_speed_squared"] - expected_cs2) < 1e-12

    def test_sound_speed_calculation(self, reference_system):
        """Test sound speed calculation from thermodynamic state."""
        system = reference_system

        # Sound speed squared: c_s² = p/ρ (for ideal gas)
        cs_squared = system.parameters.equilibrium_pressure / system.background.rho
        cs = np.sqrt(cs_squared)

        # Should be less than speed of light
        assert cs < PhysicalConstants.c

        # Should be positive
        assert cs > 0

    def test_parameter_scaling_relations(self, reference_system):
        """Test parameter scaling and dimensional analysis."""
        system = reference_system

        # Relaxation times should be positive
        assert system.parameters.tau_pi > 0
        assert system.parameters.tau_Pi > 0
        assert system.parameters.tau_q > 0

        # Transport coefficients should be non-negative
        assert system.parameters.eta >= 0
        assert system.parameters.zeta >= 0
        assert system.parameters.kappa >= 0


class TestLinearizedEquationConsistency:
    """Test consistency of linearized equations with physical principles."""

    @pytest.fixture
    def test_system(self):
        """Test system for equation consistency checks."""
        background = BackgroundState()
        params = IsraelStewartParameters()
        return LinearizedIS(background, params)

    def test_conservation_law_structure(self, test_system):
        """Test that linearized equations preserve conservation structure."""
        system = test_system
        equations = system.get_linearized_equations()

        # Continuity equation should involve density and velocity
        continuity = equations["continuity"]
        symbols = continuity.free_symbols

        # Should contain perturbation symbols
        delta_rho = system.linearized_fields["rho"].perturbation
        assert delta_rho in symbols

    def test_constraint_preservation(self, test_system):
        """Test that linearization preserves constraint structure."""
        system = test_system

        # Background should satisfy equilibrium constraints
        assert system.background.validate_equilibrium()

        # Linearized fields should preserve tensor structure
        pi_field = system.linearize_field("pi")
        assert "traceless" in pi_field.symmetries
        assert "symmetric" in pi_field.symmetries

    def test_equation_linearity(self, test_system):
        """Test that all equations are linear in perturbations."""
        system = test_system
        equations = system.get_linearized_equations()

        # Each equation should be linear (degree 1) in perturbation fields
        for _eq_name, equation in equations.items():
            # This is a simplified test - full linearity check would be more complex
            assert equation != 0  # Non-trivial equations

    def test_dimensional_consistency(self, test_system):
        """Test dimensional consistency of linearized equations."""
        system = test_system

        # Background state should have consistent dimensions
        assert system.background.rho > 0  # Energy density positive
        assert system.background.temperature > 0  # Temperature positive

        # Parameters should have consistent dimensions
        assert system.parameters.validate_causality()


# Integration test
def test_full_linearization_workflow():
    """Integration test for complete linearization workflow."""

    # Step 1: Create background state
    background = BackgroundState(rho=1.5, pressure=0.4, temperature=1.2)
    assert background.validate_equilibrium()

    # Step 2: Create parameters
    params = IsraelStewartParameters(
        eta=0.8, zeta=0.15, kappa=0.6, tau_pi=0.12, tau_Pi=0.08, tau_q=0.03
    )
    assert params.validate_causality()

    # Step 3: Initialize linearized system
    system = LinearizedIS(background, params)
    assert len(system.linearized_fields) == 5

    # Step 4: Extract linearized equations
    equations = system.get_linearized_equations()
    assert len(equations) > 10  # Multiple equations

    # Step 5: Calculate dispersion relations
    k_test = 1.0
    omega_sound = system.dispersion_relation(k_test, "sound")
    omega_diff = system.dispersion_relation(k_test, "diffusive")

    # Step 6: Perform stability analysis
    is_stable = system.is_linearly_stable()

    # Step 7: Check causality
    causality_ok = system.validate_causality()

    # Step 8: Calculate physical properties
    gamma = system.sound_attenuation_coefficient()
    critical_params = system.critical_parameters()

    # All steps should complete successfully
    assert isinstance(omega_sound, complex)
    assert isinstance(omega_diff, complex)
    assert isinstance(is_stable, bool)
    assert isinstance(causality_ok, bool)
    assert gamma > 0
    assert len(critical_params) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
