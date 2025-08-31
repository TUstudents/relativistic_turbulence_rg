"""
Unit tests for tensor algebra system
"""
import pytest
import numpy as np

from rtrg.core.tensors import Metric, LorentzTensor, IndexStructure
from rtrg.core.constants import PhysicalConstants


@pytest.mark.unit
class TestMetric:
    """Test Minkowski metric functionality"""
    
    def test_metric_initialization(self):
        """Test metric tensor initialization"""
        metric = Metric(dimension=4)
        
        assert metric.dim == 4
        assert metric.g.shape == (4, 4)
        
        # Check signature (-,+,+,+)
        expected_diag = np.array([-1, 1, 1, 1])
        np.testing.assert_array_equal(np.diag(metric.g), expected_diag)
    
    def test_metric_properties(self):
        """Test metric tensor mathematical properties"""
        metric = Metric()
        g = metric.g
        g_inv = np.linalg.inv(g)
        
        # g^μρ g_ρν = δ^μ_ν
        identity = g_inv @ g
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-14)
        
        # Metric should be diagonal
        off_diag = g - np.diag(np.diag(g))
        np.testing.assert_allclose(off_diag, np.zeros((4, 4)), atol=1e-14)
    
    def test_raise_lower_index(self, metric):
        """Test index raising and lowering"""
        # Create test vector
        v_lower = np.array([1.0, 2.0, 3.0, 4.0])  # v_μ
        
        # Raise index: v^μ = g^μν v_ν
        v_upper = metric.raise_index(v_lower, 0)
        
        # Check first component sign change due to metric signature
        assert v_upper[0] == -v_lower[0]  # g^00 = -1
        np.testing.assert_array_equal(v_upper[1:], v_lower[1:])  # g^ii = +1
        
        # Lower index back
        v_back = metric.lower_index(v_upper, 0)
        np.testing.assert_allclose(v_back, v_lower, atol=1e-14)


@pytest.mark.unit
class TestIndexStructure:
    """Test index structure management"""
    
    def test_index_structure_creation(self):
        """Test IndexStructure initialization"""
        indices = IndexStructure(
            names=['mu', 'nu'],
            types=['contravariant', 'covariant'],
            symmetries=['symmetric', 'symmetric']
        )
        
        assert indices.rank == 2
        assert indices.names == ['mu', 'nu']
        assert indices.is_symmetric()
    
    def test_invalid_index_structure(self):
        """Test validation of index structure"""
        with pytest.raises(ValueError, match="Index arrays must have same length"):
            IndexStructure(
                names=['mu'],
                types=['contravariant', 'covariant'],  # Wrong length
                symmetries=['symmetric']
            )


@pytest.mark.unit
class TestLorentzTensor:
    """Test Lorentz tensor operations"""
    
    def test_tensor_creation(self, metric):
        """Test tensor creation and basic properties"""
        # Create rank-2 tensor
        components = np.random.rand(4, 4)
        indices = IndexStructure(
            names=['mu', 'nu'],
            types=['contravariant', 'contravariant'],
            symmetries=['none', 'none']
        )
        
        tensor = LorentzTensor(components, indices, metric)
        
        assert tensor.rank == 2
        assert tensor.shape == (4, 4)
        np.testing.assert_array_equal(tensor.components, components)
    
    def test_tensor_symmetrization(self, metric):
        """Test tensor symmetrization"""
        # Create asymmetric tensor
        components = np.random.rand(4, 4)
        indices = IndexStructure(['mu', 'nu'], ['contravariant', 'contravariant'], ['none', 'none'])
        tensor = LorentzTensor(components, indices, metric)
        
        # Symmetrize
        symmetric_tensor = tensor.symmetrize()
        
        # Check symmetry
        sym_comp = symmetric_tensor.components
        np.testing.assert_allclose(sym_comp, sym_comp.T, atol=1e-14)
        
        # Check it's actually the symmetrized version
        expected = 0.5 * (components + components.T)
        np.testing.assert_allclose(sym_comp, expected, atol=1e-14)
    
    def test_tensor_trace(self, metric):
        """Test tensor trace operations"""
        # Create rank-2 tensor
        components = np.diag([1, 2, 3, 4])  # Diagonal for easy trace
        indices = IndexStructure(['mu', 'nu'], ['contravariant', 'covariant'], ['none', 'none'])
        tensor = LorentzTensor(components, indices, metric)
        
        # Take trace
        trace = tensor.trace()
        
        # Should give scalar result
        assert isinstance(trace, (int, float, complex))
        assert trace == np.trace(components)
    
    def test_four_velocity_normalization(self, metric, normalized_four_velocity):
        """Test four-velocity normalization constraint"""
        indices = IndexStructure(['mu'], ['contravariant'], ['none'])
        u_tensor = LorentzTensor(normalized_four_velocity, indices, metric)
        
        # Contract with itself: u^μ u_μ
        u_lower = u_tensor.lower_index(0)
        
        # This is a simplified test - full contraction would be more complex
        u_dot_u = np.dot(normalized_four_velocity, metric.g @ normalized_four_velocity)
        
        # Should equal -c²
        expected = -PhysicalConstants.c**2
        assert abs(u_dot_u - expected) < 1e-12
    
    def test_spatial_projection(self, metric, normalized_four_velocity):
        """Test spatial projection operations"""
        # Create test tensor
        components = np.eye(4)  # Identity tensor
        indices = IndexStructure(['mu', 'nu'], ['contravariant', 'contravariant'], ['none', 'none'])
        tensor = LorentzTensor(components, indices, metric)
        
        # Project onto spatial subspace
        projected = tensor.project_spatial(normalized_four_velocity)
        
        # Check orthogonality to four-velocity
        # This is a simplified check - full test would verify all components
        assert projected.components.shape == (4, 4)
        
        # Time-time component should be modified
        assert projected.components[0, 0] != components[0, 0]
    
    def test_index_raising_lowering(self, metric):
        """Test index raising and lowering for tensors"""
        # Create vector with covariant index
        components = np.array([1, 2, 3, 4])
        indices = IndexStructure(['mu'], ['covariant'], ['none'])
        vector = LorentzTensor(components, indices, metric)
        
        # Raise index
        vector_up = vector.raise_index(0)
        
        # Check index type changed
        assert vector_up.indices.types[0] == 'contravariant'
        
        # Check components transformed correctly
        expected = metric.g @ components
        np.testing.assert_allclose(vector_up.components, expected, atol=1e-14)
        
        # Lower index back
        vector_back = vector_up.lower_index(0)
        
        # Should recover original
        np.testing.assert_allclose(vector_back.components, components, atol=1e-14)
        assert vector_back.indices.types[0] == 'covariant'


@pytest.mark.unit 
class TestTensorConstraints:
    """Test constraint satisfaction for tensor fields"""
    
    def test_traceless_tensor(self, metric):
        """Test creation and validation of traceless tensors"""
        # Create symmetric traceless tensor (shear-like)
        pi = np.zeros((4, 4))
        pi[1, 2] = pi[2, 1] = 0.1
        pi[1, 1] = 0.05
        pi[2, 2] = -0.05  # Make traceless
        
        indices = IndexStructure(
            ['mu', 'nu'], 
            ['contravariant', 'contravariant'], 
            ['symmetric', 'symmetric']
        )
        tensor = LorentzTensor(pi, indices, metric)
        
        # Check trace
        trace = tensor.trace()
        assert abs(trace) < 1e-12
    
    def test_antisymmetric_tensor(self, metric):
        """Test antisymmetric tensor creation"""
        # Create antisymmetric tensor
        F = np.zeros((4, 4))
        F[0, 1] = 1.0
        F[1, 0] = -1.0
        F[2, 3] = 2.0  
        F[3, 2] = -2.0
        
        indices = IndexStructure(
            ['mu', 'nu'],
            ['contravariant', 'contravariant'], 
            ['none', 'none']
        )
        tensor = LorentzTensor(F, indices, metric)
        
        # Antisymmetrize
        antisym = tensor.antisymmetrize()
        
        # Check antisymmetry
        A = antisym.components
        np.testing.assert_allclose(A, -A.T, atol=1e-14)


@pytest.mark.physics
class TestPhysicalTensors:
    """Test tensors that appear in relativity"""
    
    def test_stress_energy_tensor_structure(self, metric, normalized_four_velocity):
        """Test structure of stress-energy tensor"""
        # Simple perfect fluid: T^μν = ρ u^μ u^ν + p Δ^μν
        rho = 1.0
        p = 0.3
        
        u = normalized_four_velocity
        
        # Energy-momentum tensor
        T = rho * np.outer(u, u)
        
        # Add pressure in spatial directions (simplified)
        g_inv = np.linalg.inv(metric.g)
        Delta = g_inv + np.outer(u, u) / PhysicalConstants.c**2
        T += p * Delta
        
        # Create tensor
        indices = IndexStructure(['mu', 'nu'], ['contravariant', 'contravariant'], ['symmetric', 'symmetric'])
        T_tensor = LorentzTensor(T, indices, metric)
        
        # Check energy density (T^00 component)
        assert T_tensor.components[0, 0] > 0  # Positive energy
        
        # Check symmetry
        np.testing.assert_allclose(T, T.T, atol=1e-14)
    
    def test_electromagnetic_field_tensor(self, metric):
        """Test electromagnetic field tensor properties"""
        # F^μν antisymmetric tensor
        F = np.array([
            [0, -1, -2, -3],    # F^0i = -E^i/c
            [1,  0, -4, -5],    # F^ij = ε^ijk B^k  
            [2,  4,  0, -6],
            [3,  5,  6,  0]
        ])
        
        indices = IndexStructure(['mu', 'nu'], ['contravariant', 'contravariant'], ['antisymmetric', 'antisymmetric'])
        F_tensor = LorentzTensor(F, indices, metric)
        
        # Check antisymmetry
        np.testing.assert_allclose(F, -F.T, atol=1e-14)
        
        # Check trace is zero (antisymmetric tensors are traceless)
        trace = F_tensor.trace()
        assert abs(trace) < 1e-12