"""
Unit tests for field definitions and management
"""
import pytest
import numpy as np
import sympy as sp

from rtrg.core.fields import (
    Field, ResponseField, FieldRegistry, FieldProperties,
    EnergyDensityField, FourVelocityField, ShearStressField, 
    BulkPressureField, HeatFluxField
)
from rtrg.core.constants import PhysicalConstants


@pytest.mark.unit
class TestFieldProperties:
    """Test field properties and validation"""
    
    def test_field_properties_creation(self):
        """Test FieldProperties initialization"""
        props = FieldProperties(
            name='test_field',
            latex_symbol='\\phi',
            indices=['mu'],
            index_types=['contravariant'],
            engineering_dimension=2.0,
            canonical_dimension=1.0
        )
        
        assert props.name == 'test_field'
        assert props.rank == 1
        assert props.is_vector
        assert not props.is_scalar
        assert not props.is_tensor
    
    def test_field_properties_validation(self):
        """Test validation of field properties"""
        with pytest.raises(ValueError, match="Number of indices must match"):
            FieldProperties(
                name='invalid',
                latex_symbol='\\phi',
                indices=['mu', 'nu'],
                index_types=['contravariant'],  # Wrong length
                engineering_dimension=1.0,
                canonical_dimension=1.0
            )


@pytest.mark.unit  
class TestConcreteFields:
    """Test concrete IS field implementations"""
    
    def test_energy_density_field(self, metric):
        """Test energy density field"""
        rho_field = EnergyDensityField(metric)
        
        assert rho_field.name == 'rho'
        assert rho_field.rank == 0
        assert rho_field.dimension == 4.0
        assert rho_field.properties.is_scalar
        
        # Test tensor creation
        rho_value = 1.5
        rho_tensor = rho_field.create_tensor(np.array(rho_value))
        assert rho_tensor.components == rho_value
    
    def test_four_velocity_field(self, metric, spatial_velocity):
        """Test four-velocity field"""
        u_field = FourVelocityField(metric)
        
        assert u_field.name == 'u'
        assert u_field.rank == 1
        assert u_field.dimension == 0.0  # Dimensionless
        assert u_field.properties.is_vector
        assert 'u^mu * u_mu = -c^2' in u_field.properties.constraints
        
        # Test normalization
        u_normalized = u_field.normalize(spatial_velocity)
        assert u_field.is_normalized(u_normalized)
        
        # Check Lorentz factor
        v_squared = np.dot(spatial_velocity, spatial_velocity)
        c = PhysicalConstants.c
        expected_gamma = 1.0 / np.sqrt(1 - v_squared / c**2)
        assert abs(u_normalized[0] - expected_gamma * c) < 1e-12
    
    def test_four_velocity_constraints(self, metric):
        """Test four-velocity constraint validation"""
        u_field = FourVelocityField(metric)
        
        # Valid normalized velocity
        u_valid = np.array([1.0, 0.0, 0.0, 0.0])  # Rest frame, c=1
        assert u_field.is_normalized(u_valid)
        
        # Invalid velocity (wrong normalization)
        u_invalid = np.array([1.0, 1.0, 0.0, 0.0])
        assert not u_field.is_normalized(u_invalid)
        
        # Test superluminal rejection
        v_super = np.array([1.5, 0.0, 0.0])  # v > c
        with pytest.raises(ValueError, match="exceeds speed of light"):
            u_field.normalize(v_super)
    
    def test_shear_stress_field(self, metric, sample_shear_tensor):
        """Test shear stress field"""
        pi_field = ShearStressField(metric)
        
        assert pi_field.name == 'pi'
        assert pi_field.rank == 2
        assert pi_field.dimension == 2.0
        assert pi_field.properties.is_symmetric
        assert pi_field.properties.is_traceless
        
        # Test tensor creation and constraints
        pi_tensor = pi_field.create_tensor(sample_shear_tensor)
        
        # Check symmetry (approximately - numerical precision)
        np.testing.assert_allclose(
            pi_tensor.components, 
            pi_tensor.components.T, 
            atol=1e-12
        )
        
        # Check traceless
        trace = pi_tensor.trace()
        assert abs(trace) < 1e-10
    
    def test_bulk_pressure_field(self, metric):
        """Test bulk pressure field"""
        Pi_field = BulkPressureField(metric)
        
        assert Pi_field.name == 'Pi'
        assert Pi_field.rank == 0
        assert Pi_field.dimension == 2.0
        assert Pi_field.properties.is_scalar
    
    def test_heat_flux_field(self, metric):
        """Test heat flux field"""
        q_field = HeatFluxField(metric)
        
        assert q_field.name == 'q'
        assert q_field.rank == 1
        assert q_field.dimension == 3.0
        assert q_field.properties.is_vector
        assert q_field.properties.is_spatial
        assert 'u_mu * q^mu = 0' in q_field.properties.constraints


@pytest.mark.unit
class TestResponseFields:
    """Test MSRJD response fields"""
    
    def test_response_field_creation(self, metric):
        """Test response field creation from physical field"""
        rho_field = EnergyDensityField(metric)
        tilde_rho = rho_field.response
        
        assert isinstance(tilde_rho, ResponseField)
        assert tilde_rho.name == 'tilde_rho'
        assert tilde_rho.rank == rho_field.rank
        assert tilde_rho.physical_field is rho_field
        
        # Check dimension relations
        expected_dim = -rho_field.dimension - 4
        assert tilde_rho.dimension == expected_dim
    
    def test_response_field_properties(self, metric):
        """Test response field inherits correct properties"""
        pi_field = ShearStressField(metric)
        tilde_pi = pi_field.response
        
        # Should inherit symmetries
        assert tilde_pi.properties.is_symmetric == pi_field.properties.is_symmetric
        assert tilde_pi.properties.is_traceless == pi_field.properties.is_traceless
        assert tilde_pi.properties.is_spatial == pi_field.properties.is_spatial
        
        # Should have same tensor structure
        assert tilde_pi.properties.indices == pi_field.properties.indices
        assert tilde_pi.properties.index_types == pi_field.properties.index_types


@pytest.mark.unit
class TestFieldRegistry:
    """Test field registry management"""
    
    def test_registry_creation(self):
        """Test empty registry creation"""
        registry = FieldRegistry()
        
        assert len(registry) == 0
        assert registry.list_fields() == []
        assert registry.list_response_fields() == []
    
    def test_field_registration(self, metric):
        """Test individual field registration"""
        registry = FieldRegistry()
        rho_field = EnergyDensityField(metric)
        
        registry.register_field(rho_field)
        
        assert len(registry) == 1
        assert 'rho' in registry
        assert 'tilde_rho' in registry
        assert registry.get_field('rho') is rho_field
        assert registry.get_response_field('tilde_rho') is rho_field.response
    
    def test_is_fields_creation(self, metric):
        """Test creation of all IS fields"""
        registry = FieldRegistry()
        registry.create_is_fields(metric)
        
        # Should have all 5 IS fields
        expected_fields = ['rho', 'u', 'pi', 'Pi', 'q']
        assert len(registry) == len(expected_fields)
        
        for field_name in expected_fields:
            assert field_name in registry
            assert f'tilde_{field_name}' in registry
            
        # Check field types
        assert isinstance(registry.get_field('rho'), EnergyDensityField)
        assert isinstance(registry.get_field('u'), FourVelocityField)
        assert isinstance(registry.get_field('pi'), ShearStressField)
        assert isinstance(registry.get_field('Pi'), BulkPressureField)
        assert isinstance(registry.get_field('q'), HeatFluxField)
    
    def test_field_dimensions(self, field_registry):
        """Test field dimension extraction"""
        dimensions = field_registry.field_dimensions()
        
        expected_dims = {
            'rho': 4.0,
            'u': 0.0,
            'pi': 2.0,
            'Pi': 2.0,
            'q': 3.0
        }
        
        for field_name, expected_dim in expected_dims.items():
            assert dimensions[field_name] == expected_dim


@pytest.mark.unit
class TestFieldValidation:
    """Test field constraint validation"""
    
    def test_shear_tensor_validation(self, metric):
        """Test shear tensor constraint validation"""
        pi_field = ShearStressField(metric)
        
        # Valid symmetric traceless tensor
        pi_valid = np.zeros((4, 4))
        pi_valid[1, 2] = pi_valid[2, 1] = 0.1
        pi_valid[1, 1] = 0.05
        pi_valid[2, 2] = -0.05  # Traceless
        
        assert pi_field.validate_components(pi_valid)
        
        # Invalid asymmetric tensor
        pi_invalid = np.zeros((4, 4))
        pi_invalid[1, 2] = 0.1
        pi_invalid[2, 1] = 0.2  # Different value - breaks symmetry
        
        # This would fail validation if we had full constraint checking
        # For now, create_tensor will symmetrize it
        tensor = pi_field.create_tensor(pi_invalid)
        assert abs(tensor.components[1, 2] - tensor.components[2, 1]) < 1e-12
    
    def test_four_velocity_validation(self, metric, normalized_four_velocity):
        """Test four-velocity validation"""
        u_field = FourVelocityField(metric)
        
        # Should be normalized
        assert u_field.is_normalized(normalized_four_velocity)
        
        # Test with wrong dimension
        u_wrong = np.array([1, 0, 0])  # 3D instead of 4D
        assert not u_field.is_normalized(u_wrong)


@pytest.mark.physics
class TestFieldEvolution:
    """Test field evolution equations"""
    
    def test_energy_density_evolution(self, metric):
        """Test energy density evolution equation structure"""
        rho_field = EnergyDensityField(metric)
        
        # Get symbolic evolution equation
        evolution = rho_field.evolution_equation()
        
        # Should be a sympy expression
        assert isinstance(evolution, sp.Expr)
        
        # Should contain time derivative
        t = sp.Symbol('t')
        assert evolution.has(sp.Derivative(rho_field.symbol, t))
    
    def test_shear_stress_evolution(self, metric):
        """Test shear stress evolution equation"""
        pi_field = ShearStressField(metric)
        
        # Evolution with specific parameters
        tau_pi = 0.01
        eta = 0.1
        evolution = pi_field.evolution_equation(tau_pi=tau_pi, eta=eta)
        
        assert isinstance(evolution, sp.Expr)
        
        # Should contain relaxation time and viscosity
        assert evolution.has(sp.Symbol('tau_pi'))
        assert evolution.has(sp.Symbol('eta'))
    
    def test_bulk_pressure_evolution(self, metric):
        """Test bulk pressure evolution equation"""
        Pi_field = BulkPressureField(metric)
        
        tau_Pi = 0.01
        zeta = 0.05
        evolution = Pi_field.evolution_equation(tau_Pi=tau_Pi, zeta=zeta)
        
        assert isinstance(evolution, sp.Expr)
        assert evolution.has(sp.Symbol('tau_Pi'))
        assert evolution.has(sp.Symbol('zeta'))


@pytest.mark.integration
class TestFieldInteractions:
    """Test interactions between different fields"""
    
    def test_field_registry_consistency(self, field_registry):
        """Test consistency of full field registry"""
        # All fields should have response fields
        for field_name in field_registry.list_fields():
            field = field_registry.get_field(field_name)
            response_name = f'tilde_{field_name}'
            
            assert response_name in field_registry.list_response_fields()
            
            response_field = field_registry.get_response_field(response_name)
            assert response_field.physical_field is field
    
    def test_dimensional_consistency(self, field_registry):
        """Test dimensional consistency across fields"""
        # Response fields should have correct canonical dimensions
        for field_name in field_registry.list_fields():
            field = field_registry.get_field(field_name)
            response = field.response
            
            # Canonical dimension relation
            expected_response_dim = -field.canonical_dimension - 4
            assert response.canonical_dimension == expected_response_dim
    
    def test_constraint_compatibility(self, field_registry, normalized_four_velocity):
        """Test that field constraints are mutually compatible"""
        # This is a placeholder for more sophisticated constraint checking
        # In a full implementation, we'd check orthogonality conditions, etc.
        
        u_field = field_registry.get_field('u')
        pi_field = field_registry.get_field('pi')
        q_field = field_registry.get_field('q')
        
        # These fields should all be compatible with the same 4-velocity
        assert u_field.is_normalized(normalized_four_velocity)
        
        # π^μν and q^μ should be orthogonal to u_μ (would need more implementation)
        # For now, just check they exist and have right properties
        assert pi_field.properties.is_spatial
        assert q_field.properties.is_spatial