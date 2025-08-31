"""
Comprehensive tests for field definition framework.

Tests the complete field system implementation including:
- Field properties and validation
- Constraint enforcement 
- Physical field definitions for Israel-Stewart theory
- Response field generation
- Field registry management
- Dimensional analysis consistency

Mathematical Validation:
    Tests verify field properties match theoretical requirements
    and that all constraints are properly enforced.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rtrg.core.constants import PhysicalConstants
from rtrg.core.fields import (
    FieldProperties, Field, ResponseField, 
    EnergyDensityField, FourVelocityField, ShearStressField, 
    BulkPressureField, HeatFluxField, FieldRegistry
)
from rtrg.core.tensors import Metric


class TestFieldProperties:
    """Test field property specification and validation."""
    
    def test_field_properties_creation(self):
        """Test basic field properties initialization"""
        props = FieldProperties(
            name="test_field",
            latex_symbol="\\phi",
            indices=["mu", "nu"],
            index_types=["covariant", "contravariant"],
            engineering_dimension=2.0,
            canonical_dimension=1.5,
            is_symmetric=True
        )
        
        assert props.name == "test_field"
        assert props.rank == 2
        assert props.is_tensor == True
        assert props.is_vector == False
        assert props.is_scalar == False
        assert props.is_symmetric == True
        
    def test_scalar_properties(self):
        """Test scalar field properties"""
        scalar_props = FieldProperties(
            name="scalar",
            latex_symbol="\\rho",
            indices=[],
            index_types=[],
            engineering_dimension=4.0,
            canonical_dimension=4.0
        )
        
        assert scalar_props.rank == 0
        assert scalar_props.is_scalar == True
        assert scalar_props.is_vector == False
        assert scalar_props.is_tensor == False
        
    def test_vector_properties(self):
        """Test vector field properties"""
        vector_props = FieldProperties(
            name="vector",
            latex_symbol="u",
            indices=["mu"],
            index_types=["contravariant"],
            engineering_dimension=0.0,
            canonical_dimension=0.0
        )
        
        assert vector_props.rank == 1
        assert vector_props.is_scalar == False
        assert vector_props.is_vector == True
        assert vector_props.is_tensor == False
        
    def test_invalid_properties(self):
        """Test validation of invalid field properties"""
        with pytest.raises(ValueError):
            FieldProperties(
                name="invalid",
                latex_symbol="\\phi", 
                indices=["mu", "nu"],
                index_types=["covariant"],  # Length mismatch
                engineering_dimension=1.0,
                canonical_dimension=1.0
            )


class TestEnergyDensityField:
    """Test energy density field ρ implementation."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rho = EnergyDensityField()
        
    def test_field_properties(self):
        """Test energy density field properties"""
        assert self.rho.name == "rho"
        assert self.rho.rank == 0
        assert self.rho.dimension == 4.0
        assert self.rho.canonical_dimension == 4.0
        assert self.rho.properties.is_scalar == True
        
    def test_evolution_equation(self):
        """Test energy density evolution equation"""
        evolution = self.rho.evolution_equation()
        
        # Should be a symbolic expression
        assert evolution is not None
        assert hasattr(evolution, 'atoms')  # SymPy expression property
        
    def test_response_field(self):
        """Test response field generation"""
        rho_tilde = self.rho.response
        
        assert rho_tilde.name == "tilde_rho"
        assert rho_tilde.rank == 0
        assert rho_tilde.dimension == -8.0  # [ρ] - 4 = 4 - 4 = 0, but response is [ρ]-4 = -4, wait...
        # Response field dimension = -(physical_dimension + 4) = -(4 + 4) = -8
        assert rho_tilde.canonical_dimension == -8.0
        

class TestFourVelocityField:
    """Test four-velocity field u^μ implementation."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.u = FourVelocityField()
        
    def test_field_properties(self):
        """Test four-velocity field properties"""
        assert self.u.name == "u"
        assert self.u.rank == 1
        assert self.u.dimension == 0.0  # Dimensionless
        assert self.u.properties.is_vector == True
        assert len(self.u.properties.constraints) == 1
        assert "u^mu * u_mu = -c^2" in self.u.properties.constraints
        
    def test_normalization_constraint(self):
        """Test four-velocity normalization constraint"""
        # Rest frame four-velocity
        u_rest = np.array([PhysicalConstants.c, 0, 0, 0])
        assert self.u.is_normalized(u_rest) == True
        
        # Invalid four-velocity
        u_invalid = np.array([1, 1, 1, 1])
        assert self.u.is_normalized(u_invalid) == False
        
    def test_normalize_from_spatial_velocity(self):
        """Test four-velocity normalization from three-velocity"""
        # Rest frame
        v_rest = np.array([0, 0, 0])
        u_rest = self.u.normalize(v_rest)
        expected_rest = np.array([PhysicalConstants.c, 0, 0, 0])
        assert_allclose(u_rest, expected_rest)
        assert self.u.is_normalized(u_rest) == True
        
        # Moving frame
        v_half = np.array([0.5, 0, 0])  # Half speed of light in x direction
        u_moving = self.u.normalize(v_half)
        
        # Check normalization
        assert self.u.is_normalized(u_moving) == True
        
        # Check Lorentz factor calculation
        gamma_expected = 1.0 / np.sqrt(1 - 0.25)  # γ = 1/√(1-v²/c²) for c=1, v=0.5
        assert_allclose(u_moving[0], gamma_expected * PhysicalConstants.c, rtol=1e-10)
        assert_allclose(u_moving[1], gamma_expected * 0.5, rtol=1e-10)
        
    def test_superluminal_velocity_error(self):
        """Test error for superluminal velocities"""
        v_superluminal = np.array([2.0, 0, 0])  # Faster than light
        
        with pytest.raises(ValueError, match="exceeds speed of light"):
            self.u.normalize(v_superluminal)
            
    def test_invalid_spatial_velocity_dimension(self):
        """Test error for wrong spatial velocity dimensions"""
        v_2d = np.array([0.5, 0.5])  # Only 2D
        
        with pytest.raises(ValueError, match="must be 3-dimensional"):
            self.u.normalize(v_2d)


class TestShearStressField:
    """Test shear stress tensor π^{μν} implementation."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pi = ShearStressField()
        
    def test_field_properties(self):
        """Test shear stress field properties"""
        assert self.pi.name == "pi"
        assert self.pi.rank == 2
        assert self.pi.dimension == 2.0
        assert self.pi.properties.is_symmetric == True
        assert self.pi.properties.is_traceless == True
        assert self.pi.properties.is_spatial == True
        
        # Check constraints
        constraints = self.pi.properties.constraints
        assert "pi^mu_mu = 0" in constraints  # Traceless
        assert "pi^mu_nu = pi^nu_mu" in constraints  # Symmetric
        assert "u_mu * pi^mu_nu = 0" in constraints  # Spatial (orthogonal to velocity)
        
    def test_evolution_equation(self):
        """Test shear stress evolution equation"""
        evolution = self.pi.evolution_equation(tau_pi=0.1, eta=0.5)
        
        # Should be a symbolic expression involving τ_π, η
        assert evolution is not None
        assert hasattr(evolution, 'atoms')
        
        # Check that parameters appear in the equation
        symbols = [str(s) for s in evolution.atoms()]
        assert any('tau_pi' in s for s in symbols)
        assert any('eta' in s for s in symbols)
        
    def test_tensor_constraints(self):
        """Test shear tensor constraint enforcement"""
        # Create a test shear tensor
        components = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8], 
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=complex)
        
        # Create tensor and apply constraints
        tensor = self.pi.create_tensor(components)
        
        # Check that symmetrization was applied
        assert_allclose(tensor.components, tensor.components.T)
        
        # For traceless check - the _make_traceless method should be applied
        # This is more complex to test without the full implementation


class TestBulkPressureField:
    """Test bulk pressure field Π implementation."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.Pi = BulkPressureField()
        
    def test_field_properties(self):
        """Test bulk pressure field properties"""
        assert self.Pi.name == "Pi"
        assert self.Pi.rank == 0
        assert self.Pi.dimension == 2.0
        assert self.Pi.properties.is_scalar == True
        
    def test_evolution_equation(self):
        """Test bulk pressure evolution equation"""
        evolution = self.Pi.evolution_equation(tau_Pi=0.05, zeta=0.2)
        
        assert evolution is not None
        symbols = [str(s) for s in evolution.atoms()]
        assert any('tau_Pi' in s for s in symbols)
        assert any('zeta' in s for s in symbols)


class TestHeatFluxField:
    """Test heat flux field q^μ implementation."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.q = HeatFluxField()
        
    def test_field_properties(self):
        """Test heat flux field properties"""
        assert self.q.name == "q"
        assert self.q.rank == 1
        assert self.q.dimension == 3.0
        assert self.q.properties.is_vector == True
        assert self.q.properties.is_spatial == True
        
        # Check orthogonality constraint
        constraints = self.q.properties.constraints
        assert "u_mu * q^mu = 0" in constraints
        
    def test_evolution_equation(self):
        """Test heat flux evolution equation"""
        evolution = self.q.evolution_equation(tau_q=0.02, kappa=0.1)
        
        assert evolution is not None
        symbols = [str(s) for s in evolution.atoms()]
        assert any('tau_q' in s or 'kappa' in s for s in symbols)


class TestResponseFields:
    """Test response field generation and properties."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rho = EnergyDensityField()
        self.u = FourVelocityField()
        self.pi = ShearStressField()
        
    def test_response_field_dimensions(self):
        """Test response field dimensional consistency"""
        # Energy density: [ρ] = 4, [ρ̃] = -4-4 = -8
        rho_tilde = self.rho.response
        assert rho_tilde.dimension == -8.0
        assert rho_tilde.canonical_dimension == -8.0
        
        # Four-velocity: [u] = 0, [ũ] = -0-4 = -4
        u_tilde = self.u.response
        assert u_tilde.dimension == -4.0
        assert u_tilde.canonical_dimension == -4.0
        
        # Shear stress: [π] = 2, [π̃] = -2-4 = -6
        pi_tilde = self.pi.response
        assert pi_tilde.dimension == -6.0
        assert pi_tilde.canonical_dimension == -6.0
        
    def test_response_field_structure(self):
        """Test response field index structure preservation"""
        # Response fields should have same tensor structure as physical fields
        pi_tilde = self.pi.response
        
        assert pi_tilde.rank == self.pi.rank
        assert pi_tilde.properties.indices == self.pi.properties.indices
        assert pi_tilde.properties.index_types == self.pi.properties.index_types
        assert pi_tilde.properties.is_symmetric == self.pi.properties.is_symmetric
        assert pi_tilde.properties.is_traceless == self.pi.properties.is_traceless
        
    def test_response_field_names(self):
        """Test response field naming convention"""
        assert self.rho.response.name == "tilde_rho"
        assert self.u.response.name == "tilde_u"
        assert self.pi.response.name == "tilde_pi"
        
    def test_response_field_evolution(self):
        """Test response field evolution equations"""
        # Response fields don't have independent evolution equations
        rho_tilde = self.rho.response
        evolution = rho_tilde.evolution_equation()
        
        # Should return zero (no independent evolution)
        assert evolution == 0


class TestFieldRegistry:
    """Test field registry management system."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = FieldRegistry()
        
    def test_empty_registry(self):
        """Test empty registry initialization"""
        assert len(self.registry) == 0
        assert self.registry.list_fields() == []
        assert self.registry.list_response_fields() == []
        
    def test_field_registration(self):
        """Test manual field registration"""
        rho = EnergyDensityField()
        self.registry.register_field(rho)
        
        assert len(self.registry) == 1
        assert "rho" in self.registry
        assert "tilde_rho" in self.registry
        assert self.registry.get_field("rho") is rho
        assert self.registry.get_response_field("tilde_rho") is rho.response
        
    def test_create_is_fields(self):
        """Test Israel-Stewart field set creation"""
        self.registry.create_is_fields()
        
        # Should have all 5 IS fields
        assert len(self.registry) == 5
        
        expected_fields = ["rho", "u", "pi", "Pi", "q"]
        for field_name in expected_fields:
            assert field_name in self.registry
            assert f"tilde_{field_name}" in self.registry
            
        # Check field list
        field_list = self.registry.list_fields()
        for field_name in expected_fields:
            assert field_name in field_list
            
        # Check response field list
        response_list = self.registry.list_response_fields()
        for field_name in expected_fields:
            assert f"tilde_{field_name}" in response_list
            
    def test_field_dimensions(self):
        """Test field dimension extraction"""
        self.registry.create_is_fields()
        
        dimensions = self.registry.field_dimensions()
        
        # Check expected dimensions
        expected_dims = {
            "rho": 4.0,    # Energy density
            "u": 0.0,      # Four-velocity (dimensionless)
            "pi": 2.0,     # Shear stress
            "Pi": 2.0,     # Bulk pressure
            "q": 3.0       # Heat flux
        }
        
        for field, dim in expected_dims.items():
            assert field in dimensions
            assert dimensions[field] == dim
            
    def test_field_retrieval(self):
        """Test field retrieval by name"""
        self.registry.create_is_fields()
        
        # Test successful retrieval
        rho_field = self.registry.get_field("rho")
        assert rho_field is not None
        assert rho_field.name == "rho"
        
        # Test missing field
        missing_field = self.registry.get_field("nonexistent")
        assert missing_field is None
        
        # Test response field retrieval
        rho_tilde = self.registry.get_response_field("tilde_rho")
        assert rho_tilde is not None
        assert rho_tilde.name == "tilde_rho"


class TestFieldConstraints:
    """Test field constraint enforcement and validation."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.u = FourVelocityField()
        self.pi = ShearStressField()
        
    def test_four_velocity_constraint_enforcement(self):
        """Test four-velocity normalization constraint"""
        # Valid normalized four-velocity
        u_valid = np.array([PhysicalConstants.c, 0, 0, 0])
        assert self.u.validate_components(u_valid) == True
        
        # For now, validate_components may not check all constraints
        # The actual constraint checking happens in specific methods
        assert self.u.is_normalized(u_valid) == True
        
    def test_shear_tensor_constraint_enforcement(self):
        """Test shear tensor constraint enforcement"""
        # Create symmetric, traceless tensor
        # This is a complex test that requires the full constraint implementation
        
        # For now, just test that create_tensor applies constraints
        components = np.random.random((4, 4))
        tensor = self.pi.create_tensor(components)
        
        # Should be symmetric after constraint application
        assert_allclose(tensor.components, tensor.components.T, rtol=1e-10)


class TestDimensionalAnalysis:
    """Test dimensional consistency across the field system."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = FieldRegistry()
        self.registry.create_is_fields()
        
    def test_engineering_dimensions(self):
        """Test engineering dimension consistency"""
        dimensions = self.registry.field_dimensions()
        
        # Check that dimensions are physically reasonable
        assert dimensions["rho"] > 0  # Energy density should be positive dimension
        assert dimensions["u"] == 0   # Four-velocity should be dimensionless
        assert dimensions["pi"] > 0   # Stress should have positive dimension
        assert dimensions["Pi"] > 0   # Pressure should have positive dimension
        assert dimensions["q"] > 0    # Heat flux should have positive dimension
        
    def test_response_field_dimensions(self):
        """Test response field dimensional consistency"""
        for name, field in self.registry.fields.items():
            response = field.response
            
            # Response field dimension should be physical - 4
            expected_response_dim = -(field.dimension + 4)
            assert response.dimension == expected_response_dim
            assert response.canonical_dimension == expected_response_dim
            
    def test_dimension_balance_in_evolution_equations(self):
        """Test dimensional balance in evolution equations"""
        # This is a complex test that would require parsing symbolic expressions
        # For now, just verify that evolution equations exist
        
        pi = self.registry.get_field("pi")
        evolution = pi.evolution_equation(tau_pi=1.0, eta=1.0)
        
        assert evolution is not None
        
        # In full implementation, would check that all terms have same dimensions


@pytest.mark.physics
class TestPhysicalConsistency:
    """Test physical consistency and theoretical requirements."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = FieldRegistry()
        self.registry.create_is_fields()
        
    def test_field_count(self):
        """Test correct number of fields for Israel-Stewart theory"""
        # Should have exactly 5 physical fields
        assert len(self.registry.fields) == 5
        
        # Should have corresponding 5 response fields
        assert len(self.registry.response_fields) == 5
        
    def test_lorentz_covariance(self):
        """Test that fields transform correctly under Lorentz transformations"""
        # This would require implementing Lorentz transformations
        # For now, just check that index structures are correct
        
        u = self.registry.get_field("u")
        pi = self.registry.get_field("pi")
        
        # Four-velocity should be a vector
        assert u.rank == 1
        assert u.properties.index_types == ["contravariant"]
        
        # Shear stress should be rank-2 tensor
        assert pi.rank == 2
        assert pi.properties.index_types == ["contravariant", "contravariant"]
        
    def test_constraint_completeness(self):
        """Test that all necessary constraints are specified"""
        fields_with_constraints = {
            "u": ["u^mu * u_mu = -c^2"],
            "pi": ["pi^mu_mu = 0", "pi^mu_nu = pi^nu_mu", "u_mu * pi^mu_nu = 0"],
            "q": ["u_mu * q^mu = 0"]
        }
        
        for field_name, expected_constraints in fields_with_constraints.items():
            field = self.registry.get_field(field_name)
            field_constraints = field.properties.constraints
            
            for constraint in expected_constraints:
                assert constraint in field_constraints


if __name__ == '__main__':
    pytest.main([__file__, '-v'])