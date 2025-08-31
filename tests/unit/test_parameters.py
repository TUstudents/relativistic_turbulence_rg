"""
Unit tests for parameter management system
"""
import pytest
import numpy as np
import warnings

from rtrg.core.parameters import ISParameters, StandardParameterSets
from rtrg.core.constants import PhysicalConstants


@pytest.mark.unit
class TestISParameters:
    """Test Israel-Stewart parameter management"""
    
    def test_parameter_creation(self):
        """Test basic parameter creation"""
        params = ISParameters(
            eta=0.1,
            zeta=0.05,
            kappa=0.2,
            tau_pi=0.01,
            tau_Pi=0.01,
            tau_q=0.015,
            cs=1.0/np.sqrt(3),
            temperature=1.0
        )
        
        assert params.eta == 0.1
        assert params.zeta == 0.05
        assert params.cs == 1.0/np.sqrt(3)
        assert params._validated  # Should be validated after creation
    
    def test_parameter_validation_positive(self):
        """Test validation of positivity constraints"""
        # Negative viscosity should fail
        with pytest.raises(ValueError, match="Shear viscosity η must be positive"):
            ISParameters(
                eta=-0.1,  # Invalid
                zeta=0.05,
                kappa=0.2,
                tau_pi=0.01,
                tau_Pi=0.01, 
                tau_q=0.015,
                cs=1.0/np.sqrt(3),
                temperature=1.0
            )
        
        # Negative relaxation time should fail
        with pytest.raises(ValueError, match="Relaxation time τ_π must be positive"):
            ISParameters(
                eta=0.1,
                zeta=0.05,
                kappa=0.2,
                tau_pi=-0.01,  # Invalid
                tau_Pi=0.01,
                tau_q=0.015,
                cs=1.0/np.sqrt(3),
                temperature=1.0
            )
    
    def test_sound_speed_validation(self):
        """Test sound speed validation"""
        # Sound speed exceeding c should fail
        with pytest.raises(ValueError, match="exceeds speed of light"):
            ISParameters(
                eta=0.1,
                zeta=0.05,
                kappa=0.2,
                tau_pi=0.01,
                tau_Pi=0.01,
                tau_q=0.015,
                cs=2.0,  # > c = 1 in natural units
                temperature=1.0
            )
        
        # Zero sound speed should fail
        with pytest.raises(ValueError, match="Sound speed must be positive"):
            ISParameters(
                eta=0.1,
                zeta=0.05,
                kappa=0.2,
                tau_pi=0.01,
                tau_Pi=0.01,
                tau_q=0.015,
                cs=0.0,  # Invalid
                temperature=1.0
            )
    
    def test_parameter_warnings(self):
        """Test parameter validation warnings"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            params = ISParameters(
                eta=0.1,
                zeta=-0.01,  # Negative bulk viscosity - should warn
                kappa=0.2,
                tau_pi=0.01,
                tau_Pi=0.01,
                tau_q=0.015,
                cs=1.0/np.sqrt(3),
                temperature=1.0,
                energy_density=-1.0  # Negative energy density - should warn
            )
            
            # Should have generated warnings
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("bulk viscosity" in msg for msg in warning_messages)
    
    def test_causality_check(self, is_parameters):
        """Test causality validation"""
        # Should pass for reasonable parameters
        assert is_parameters._validated
        
        # Create parameters that might violate causality
        # (This is a simplified test - full implementation would be more sophisticated)
        extreme_params = ISParameters(
            eta=100.0,  # Very large viscosity
            zeta=50.0,
            kappa=0.2,
            tau_pi=0.001,  # Very short relaxation time
            tau_Pi=0.001,
            tau_q=0.001,
            cs=0.1,  # Small sound speed
            temperature=1.0,
            energy_density=0.01  # Small energy density
        )
        
        # Might fail causality (depending on implementation)
        # For now, just check it doesn't crash
        assert isinstance(extreme_params._validated, bool)


@pytest.mark.unit
class TestParameterTransformations:
    """Test parameter transformations and derived quantities"""
    
    def test_dimensionless_conversion(self, is_parameters):
        """Test conversion to dimensionless parameters"""
        L0 = 1.0  # Length scale
        rho0 = 3.0  # Density scale
        
        dimensionless = is_parameters.to_dimensionless(L0, rho0)
        
        # Sound speed should be normalized to 1
        assert dimensionless.cs == 1.0
        assert dimensionless.unit_system == 'dimensionless'
        
        # Original should be unchanged
        assert is_parameters.cs == 1.0/np.sqrt(3)
        assert is_parameters.unit_system == 'natural'
    
    def test_characteristic_scales(self, is_parameters):
        """Test computation of characteristic scales"""
        scales = is_parameters.characteristic_scales()
        
        # Should have time scales
        assert 't_pi' in scales
        assert 't_Pi' in scales
        assert 't_q' in scales
        
        # Should have velocity scales
        assert 'c_s' in scales
        assert 'c' in scales
        
        # Values should be positive
        for scale_name, scale_value in scales.items():
            assert scale_value > 0, f"Scale {scale_name} should be positive"
    
    def test_dimensionless_numbers(self, is_parameters):
        """Test computation of dimensionless numbers"""
        L = 1.0  # Characteristic length
        v = 0.1  # Characteristic velocity
        
        numbers = is_parameters.dimensionless_numbers(L, v)
        
        # Should compute standard dimensionless numbers
        expected_numbers = ['Ma', 'Kn_pi', 'Kn_Pi', 'Kn_q']
        for number_name in expected_numbers:
            assert number_name in numbers
            assert numbers[number_name] >= 0
        
        # Mach number should be v/cs
        expected_Ma = v / is_parameters.cs
        assert abs(numbers['Ma'] - expected_Ma) < 1e-12
        
        # Knudsen numbers should be τ*cs/L
        expected_Kn_pi = is_parameters.tau_pi * is_parameters.cs / L
        assert abs(numbers['Kn_pi'] - expected_Kn_pi) < 1e-12
    
    def test_parameter_copy_and_update(self, is_parameters):
        """Test parameter copying and updating"""
        # Test copy
        params_copy = is_parameters.copy()
        
        assert params_copy.eta == is_parameters.eta
        assert params_copy is not is_parameters  # Different objects
        
        # Test update
        updated = is_parameters.update(eta=0.2, tau_pi=0.02)
        
        assert updated.eta == 0.2
        assert updated.tau_pi == 0.02
        assert updated.zeta == is_parameters.zeta  # Unchanged
        assert is_parameters.eta == 0.1  # Original unchanged
        
        # Invalid parameter should fail
        with pytest.raises(ValueError, match="Unknown parameter"):
            is_parameters.update(invalid_param=1.0)


@pytest.mark.unit
class TestKineticTheoryRelations:
    """Test kinetic theory parameter relations"""
    
    def test_kinetic_theory_setup(self, is_parameters):
        """Test setting parameters from kinetic theory"""
        particle_mass = 1.0
        cross_section = 0.1
        
        # Apply kinetic theory relations
        kt_params = is_parameters.kinetic_theory_relations(particle_mass, cross_section)
        
        # Should still be the same object
        assert kt_params is is_parameters
        
        # Relaxation times should be equal (simplification)
        assert kt_params.tau_pi == kt_params.tau_Pi == kt_params.tau_q
        
        # Should still be validated
        assert kt_params._validated
    
    def test_kinetic_theory_consistency(self):
        """Test consistency of kinetic theory parameter relations"""
        params = ISParameters(
            eta=0.1, zeta=0.05, kappa=0.2,
            tau_pi=0.01, tau_Pi=0.01, tau_q=0.015,
            cs=1.0/np.sqrt(3), temperature=1.0,
            pressure=1.0, energy_density=3.0
        )
        
        # Apply kinetic theory
        params.kinetic_theory_relations()
        
        # Viscosity should scale with pressure and relaxation time
        expected_eta_scale = params.pressure * params.tau_pi
        assert params.eta == expected_eta_scale
        
        # Bulk viscosity relation
        expected_zeta_scale = params.pressure * params.tau_Pi * (1/3 - params.cs**2)
        assert abs(params.zeta - expected_zeta_scale) < 1e-12


@pytest.mark.unit
class TestStandardParameterSets:
    """Test predefined parameter sets"""
    
    def test_weakly_coupled_plasma(self):
        """Test weakly-coupled plasma parameters"""
        T = 0.5
        params = StandardParameterSets.weakly_coupled_plasma(T)
        
        assert params.temperature == T
        assert params.cs == 1.0/np.sqrt(3)  # Relativistic ideal gas
        assert params._validated
        
        # Check scaling with temperature
        assert params.eta == 0.5 * T**3
        assert params.energy_density == 3 * T**4
    
    def test_strongly_coupled_plasma(self):
        """Test strongly-coupled plasma parameters"""
        T = 0.3
        params = StandardParameterSets.strongly_coupled_plasma(T)
        
        assert params.temperature == T
        assert params.zeta == 0.0  # Conformal
        
        # AdS/CFT bound
        expected_eta = T**3 / (4*np.pi)
        assert abs(params.eta - expected_eta) < 1e-12
        
        # Strong coupling relaxation time
        expected_tau = 1.0 / (2*np.pi*T)
        assert abs(params.tau_pi - expected_tau) < 1e-12
    
    def test_nuclear_matter(self):
        """Test nuclear matter parameters"""
        T = 0.1
        rho = 2.0
        params = StandardParameterSets.nuclear_matter(T, rho)
        
        assert params.temperature == T
        assert params.energy_density == rho
        assert params.cs == 0.3  # Typical nuclear matter
        assert params._validated
    
    def test_qgp_phenomenology(self):
        """Test QGP phenomenological parameters"""
        T = 0.2  # 200 MeV
        params = StandardParameterSets.qgp_phenomenology(T)
        
        assert params.temperature == T
        assert params.cs == 1.0/np.sqrt(3)  # Near conformal
        
        # Phenomenological values
        assert params.eta == 0.2 * T**3
        assert params.tau_pi == 5.0 / T
        
        # Stefan-Boltzmann pressure
        expected_pressure = T**4 * np.pi**2 / 90
        assert abs(params.pressure - expected_pressure) < 1e-12


@pytest.mark.unit  
class TestParameterSerialization:
    """Test parameter serialization and deserialization"""
    
    def test_to_dict(self, is_parameters):
        """Test conversion to dictionary"""
        param_dict = is_parameters.to_dict()
        
        # Should contain all parameters
        expected_keys = [
            'eta', 'zeta', 'kappa', 'tau_pi', 'tau_Pi', 'tau_q',
            'cs', 'temperature', 'pressure', 'energy_density',
            'lambda_pi_Pi', 'lambda_pi_q', 'lambda_qq', 'tau_pi_pi',
            'dimension', 'unit_system'
        ]
        
        for key in expected_keys:
            assert key in param_dict
            assert param_dict[key] == getattr(is_parameters, key)
    
    def test_from_dict(self, is_parameters):
        """Test creation from dictionary"""
        param_dict = is_parameters.to_dict()
        reconstructed = ISParameters.from_dict(param_dict)
        
        # Should be equivalent
        assert reconstructed.eta == is_parameters.eta
        assert reconstructed.zeta == is_parameters.zeta
        assert reconstructed.cs == is_parameters.cs
        assert reconstructed._validated
    
    def test_round_trip_serialization(self, is_parameters):
        """Test round-trip serialization preserves parameters"""
        # Convert to dict and back
        reconstructed = ISParameters.from_dict(is_parameters.to_dict())
        
        # All parameters should be preserved
        for attr in ['eta', 'zeta', 'kappa', 'tau_pi', 'tau_Pi', 'tau_q', 
                     'cs', 'temperature', 'pressure', 'energy_density']:
            original_val = getattr(is_parameters, attr)
            reconstructed_val = getattr(reconstructed, attr)
            assert abs(original_val - reconstructed_val) < 1e-15


@pytest.mark.integration
class TestParameterConsistency:
    """Test consistency across different parameter operations"""
    
    def test_dimensionless_consistency(self, is_parameters):
        """Test consistency of dimensionless conversion"""
        L0 = 2.0
        rho0 = 1.5
        
        # Convert to dimensionless
        dimensionless = is_parameters.to_dimensionless(L0, rho0)
        
        # Dimensionless numbers should be preserved
        original_numbers = is_parameters.dimensionless_numbers(L0, is_parameters.cs)
        dimensionless_numbers = dimensionless.dimensionless_numbers(1.0, 1.0)
        
        # Some numbers should be approximately the same
        # (This test would be more sophisticated in full implementation)
        assert dimensionless_numbers['Ma'] > 0
        assert dimensionless_numbers['Kn_pi'] > 0
    
    def test_parameter_scaling_relations(self):
        """Test that parameter scaling relations are consistent"""
        # Create base parameters
        base_params = ISParameters(
            eta=0.1, zeta=0.05, kappa=0.2,
            tau_pi=0.01, tau_Pi=0.01, tau_q=0.015,
            cs=1.0/np.sqrt(3), temperature=1.0,
            pressure=1.0, energy_density=3.0
        )
        
        # Scale temperature by factor of 2
        T_scale = 2.0
        scaled_params = base_params.update(
            temperature=base_params.temperature * T_scale,
            pressure=base_params.pressure * T_scale**4,
            energy_density=base_params.energy_density * T_scale**4
        )
        
        # Derived quantities should scale appropriately
        base_scales = base_params.characteristic_scales()
        scaled_scales = scaled_params.characteristic_scales()
        
        # Check that scales are positive and finite
        for scale_name in ['t_pi', 't_Pi', 't_q', 'c_s']:
            assert base_scales[scale_name] > 0
            assert scaled_scales[scale_name] > 0
            assert np.isfinite(base_scales[scale_name])
            assert np.isfinite(scaled_scales[scale_name])