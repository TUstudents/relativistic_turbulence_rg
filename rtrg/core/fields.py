"""
Field definitions and constraint management for relativistic Israel-Stewart hydrodynamics.

This module implements the complete field structure for Israel-Stewart (IS) theory,
providing both physical dynamical fields and their corresponding response fields
required for the Martin-Siggia-Rose-Janssen-De Dominicis (MSRJD) path integral
formulation of stochastic hydrodynamics.

Theoretical Background:
    Israel-Stewart theory extends ideal relativistic hydrodynamics by treating
    dissipative fluxes (shear stress, bulk pressure, heat flux) as independent
    dynamical variables with finite relaxation times. This resolves the causality
    and stability issues present in first-order theories (Eckart, Landau-Lifshitz).

Field Content:
    Physical Fields (5):
        - ρ(x^μ): Energy density (scalar)
        - u^μ(x^ν): Four-velocity (vector, constrained: u^μu_μ = -c²)
        - π^μν(x^α): Shear stress (rank-2, symmetric, traceless, spatial)
        - Π(x^μ): Bulk viscous pressure (scalar)
        - q^μ(x^ν): Heat flux (vector, spatial: u_μq^μ = 0)
    
    Response Fields (5):
        - ρ̃, ũ_μ, π̃_μν, Π̃, q̃_μ: MSRJD response fields with dimensions [-4-Δ]

Mathematical Framework:
    - All fields transform as Lorentz tensors under coordinate transformations
    - Constraints enforced via Lagrange multipliers or projection operators  
    - Engineering dimensions follow natural unit conventions (c = ℏ = k_B = 1)
    - Response fields enable functional derivative calculations in path integrals

References:
    - Israel, W. & Stewart, J.M. Ann. Physics 118, 341 (1979)
    - Kovtun, P. et al. J. High Energy Phys. 10, 064 (2011)  
    - Romatschke, P. Class. Quantum Grav. 27, 025006 (2010)
"""
import numpy as np
import sympy as sp
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .tensors import LorentzTensor, IndexStructure, Metric
from .constants import PhysicalConstants


@dataclass
class FieldProperties:
    """
    Comprehensive field property specification for relativistic tensor fields.
    
    Encapsulates all mathematical and physical properties needed to define
    tensor fields in Israel-Stewart theory, including Lorentz structure,
    dimensional analysis, symmetry properties, and constraint specifications.
    
    Dimensional Analysis:
        Engineering dimensions follow natural units where [length] = [time] = [mass]^{-1}.
        Canonical dimensions determine RG scaling behavior under Wilson renormalization.
        
    Constraint Types:
        - Algebraic: Relationships between field components (e.g., tracelessness)
        - Geometric: Orthogonality conditions (e.g., spatial projection)  
        - Normalization: Fixed magnitude conditions (e.g., four-velocity norm)
        
    Attributes:
        name: Field identifier used in code and symbolic expressions
        latex_symbol: LaTeX representation for mathematical typesetting
        indices: Lorentz index names in standard notation (['mu', 'nu', ...])
        index_types: Covariance specification for each index ('covariant'/'contravariant')
        engineering_dimension: Physical dimension in natural units [mass^n]
        canonical_dimension: Scaling dimension for renormalization group analysis
        is_symmetric: Whether tensor is symmetric under index permutation
        is_traceless: Whether tensor trace vanishes (e.g., T^μ_μ = 0)
        is_spatial: Whether tensor is orthogonal to timelike four-velocity
        constraints: List of mathematical constraint descriptions
        
    Examples:
        >>> # Four-velocity properties
        >>> u_props = FieldProperties(
        ...     name='u', latex_symbol='u', indices=['mu'], index_types=['contravariant'],
        ...     engineering_dimension=0.0, canonical_dimension=0.0,
        ...     constraints=['u^mu * u_mu = -c^2']
        ... )
        >>>
        >>> # Shear stress properties  
        >>> pi_props = FieldProperties(
        ...     name='pi', latex_symbol='\\pi', indices=['mu', 'nu'], 
        ...     index_types=['contravariant', 'contravariant'],
        ...     engineering_dimension=2.0, canonical_dimension=2.0,
        ...     is_symmetric=True, is_traceless=True, is_spatial=True,
        ...     constraints=['pi^mu_mu = 0', 'pi^mu_nu = pi^nu_mu', 'u_mu * pi^mu_nu = 0']
        ... )
    """
    name: str
    latex_symbol: str
    indices: List[str]
    index_types: List[str] 
    engineering_dimension: float
    canonical_dimension: float
    is_symmetric: bool = False
    is_traceless: bool = False
    is_spatial: bool = False
    constraints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate field properties"""
        if len(self.indices) != len(self.index_types):
            raise ValueError("Number of indices must match number of index types")
    
    @property
    def rank(self) -> int:
        """Tensor rank of the field"""
        return len(self.indices)
    
    @property  
    def is_scalar(self) -> bool:
        """Check if field is a scalar (rank 0)"""
        return self.rank == 0
    
    @property
    def is_vector(self) -> bool:
        """Check if field is a vector (rank 1)"""
        return self.rank == 1
    
    @property
    def is_tensor(self) -> bool:
        """Check if field is a tensor (rank >= 2)"""
        return self.rank >= 2


class Field(ABC):
    """Abstract base class for fields in the theory"""
    
    def __init__(self, properties: FieldProperties, metric: Optional[Metric] = None):
        """Initialize field
        
        Args:
            properties: Field properties and constraints
            metric: Spacetime metric (default Minkowski)
        """
        self.properties = properties
        self.metric = metric or Metric()
        
        # Create symbolic representation
        self.symbol = sp.Symbol(properties.name)
        if properties.indices:
            # Create indexed symbol for non-scalars
            index_symbols = [sp.Symbol(idx) for idx in properties.indices]
            self.symbol = sp.IndexedBase(properties.name)[tuple(index_symbols)]
        
        # Response field for MSRJD formalism
        self._response_field = None
    
    @property
    def name(self) -> str:
        """Field name"""
        return self.properties.name
    
    @property
    def rank(self) -> int:
        """Tensor rank"""
        return self.properties.rank
    
    @property
    def dimension(self) -> float:
        """Engineering dimension"""
        return self.properties.engineering_dimension
    
    @property
    def canonical_dimension(self) -> float:
        """Canonical dimension for RG scaling"""
        return self.properties.canonical_dimension
    
    @property
    def response(self) -> 'ResponseField':
        """Get response field for MSRJD formalism"""
        if self._response_field is None:
            self._response_field = ResponseField(self)
        return self._response_field
    
    def create_tensor(self, components: np.ndarray) -> LorentzTensor:
        """Create LorentzTensor from components
        
        Args:
            components: Field components as numpy array
            
        Returns:
            LorentzTensor with proper index structure
        """
        index_structure = IndexStructure(
            names=self.properties.indices,
            types=self.properties.index_types,
            symmetries=['symmetric' if self.properties.is_symmetric else 'none'] * self.rank
        )
        
        tensor = LorentzTensor(components, index_structure, self.metric)
        
        # Apply constraints
        if self.properties.is_traceless and self.rank >= 2:
            tensor = self._make_traceless(tensor)
        
        return tensor
    
    def validate_components(self, components: np.ndarray) -> bool:
        """Validate field components satisfy constraints
        
        Args:
            components: Field components to validate
            
        Returns:
            True if components satisfy all constraints
        """
        tensor = self.create_tensor(components)
        
        # Check symmetry
        if self.properties.is_symmetric:
            if not self._check_symmetry(tensor):
                return False
        
        # Check tracelessness
        if self.properties.is_traceless:
            if not self._check_traceless(tensor):
                return False
        
        # Check spatial orthogonality (requires velocity context)
        # This would need to be implemented with a velocity field
        
        return True
    
    def _make_traceless(self, tensor: LorentzTensor) -> LorentzTensor:
        """Make tensor traceless by subtracting trace part"""
        if tensor.rank < 2:
            return tensor
        
        # For rank-2 tensor: T^μν - (1/d)g^μν T^α_α
        if tensor.rank == 2:
            trace = tensor.trace()
            g_inv = np.linalg.inv(self.metric.g) 
            trace_part = trace / self.metric.dim * g_inv
            
            # Subtract trace part
            result_components = tensor.components - trace_part
            return LorentzTensor(result_components, tensor.indices, tensor.metric)
        
        # Higher rank tensors would need more sophisticated treatment
        return tensor
    
    def _check_symmetry(self, tensor: LorentzTensor) -> bool:
        """Check if tensor is symmetric"""
        if tensor.rank < 2:
            return True
        
        # Check all index pairs for symmetry
        for i in range(tensor.rank):
            for j in range(i + 1, tensor.rank):
                # Swap indices i and j
                axes = list(range(tensor.rank))
                axes[i], axes[j] = axes[j], axes[i]
                swapped = tensor.components.transpose(axes)
                
                if not np.allclose(tensor.components, swapped, rtol=1e-12):
                    return False
        
        return True
    
    def _check_traceless(self, tensor: LorentzTensor) -> bool:
        """Check if tensor is traceless"""
        if tensor.rank < 2:
            return True
        
        # Check trace is zero (within numerical precision)
        trace = tensor.trace()
        if isinstance(trace, (int, float, complex)):
            return abs(trace) < 1e-12
        
        # For higher rank, check all possible traces
        return np.allclose(trace.components, 0, atol=1e-12)
    
    @abstractmethod
    def evolution_equation(self, **kwargs) -> sp.Expr:
        """Return the evolution equation for this field"""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ResponseField(Field):
    """Response field for MSRJD path integral formulation"""
    
    def __init__(self, physical_field: Field):
        """Initialize response field
        
        Args:
            physical_field: The physical field this responds to
        """
        # Response field has same tensor structure but different dimensions
        response_properties = FieldProperties(
            name=f"tilde_{physical_field.name}",
            latex_symbol=f"\\tilde{{{physical_field.properties.latex_symbol}}}",
            indices=physical_field.properties.indices.copy(),
            index_types=physical_field.properties.index_types.copy(),
            engineering_dimension=-physical_field.properties.engineering_dimension - 4,
            canonical_dimension=-physical_field.properties.canonical_dimension - 4,
            is_symmetric=physical_field.properties.is_symmetric,
            is_traceless=physical_field.properties.is_traceless,
            is_spatial=physical_field.properties.is_spatial
        )
        
        super().__init__(response_properties, physical_field.metric)
        self.physical_field = physical_field
    
    def evolution_equation(self, **kwargs) -> sp.Expr:
        """Response fields don't have independent evolution equations"""
        return sp.Integer(0)


# Concrete field implementations for Israel-Stewart theory

class EnergyDensityField(Field):
    """Energy density ρ (scalar field)"""
    
    def __init__(self, metric: Optional[Metric] = None):
        properties = FieldProperties(
            name='rho',
            latex_symbol='\\rho',
            indices=[],
            index_types=[],
            engineering_dimension=4.0,  # [ρ] = M L^{-3} ~ M^4 in natural units
            canonical_dimension=4.0
        )
        super().__init__(properties, metric)
    
    def evolution_equation(self, **kwargs) -> sp.Expr:
        """∂_t ρ + ∂_i(ρ u^i) = 0 (continuity equation)"""
        t, x = sp.symbols('t x')  
        u = sp.Function('u')
        
        # Simplified symbolic form - full implementation would be more complex
        return sp.Derivative(self.symbol, t) + sp.Derivative(self.symbol * u(x), x)


class FourVelocityField(Field):
    """Four-velocity u^μ (vector field with constraint)"""
    
    def __init__(self, metric: Optional[Metric] = None):
        properties = FieldProperties(
            name='u',
            latex_symbol='u',
            indices=['mu'],
            index_types=['contravariant'],
            engineering_dimension=0.0,  # Dimensionless
            canonical_dimension=0.0,
            constraints=['u^mu * u_mu = -c^2']  # Normalization constraint
        )
        super().__init__(properties, metric)
    
    def evolution_equation(self, **kwargs) -> sp.Expr:
        """Evolution from momentum conservation"""
        # This would be derived from ∂_μ T^{μν} = 0
        return sp.Integer(0)  # Placeholder
    
    def is_normalized(self, components: np.ndarray) -> bool:
        """Check if four-velocity satisfies normalization constraint"""
        if len(components) != self.metric.dim:
            return False
        
        # u^μ u_μ = -c²
        u_dot_u = np.dot(components, self.metric.g @ components)
        c_squared = PhysicalConstants.c**2
        
        return abs(u_dot_u + c_squared) < 1e-12
    
    def normalize(self, spatial_velocity: np.ndarray) -> np.ndarray:
        """
        Construct normalized four-velocity from three-velocity using Lorentz transformations.
        
        Computes the properly normalized four-velocity u^μ = γ(c, v^i) from a given
        three-velocity v^i, ensuring satisfaction of the fundamental constraint
        u^μu_μ = -c². The calculation uses special relativistic kinematics.
        
        Mathematical Construction:
            Given spatial velocity v^i = (v^x, v^y, v^z), the four-velocity is:
            
            u^0 = γc = c/√(1 - v²/c²)
            u^i = γv^i = v^i/√(1 - v²/c²)
            
            where v² = δ_ij v^i v^j is the spatial velocity squared and γ is the
            Lorentz factor. The normalization u^μu_μ = -c² is automatically satisfied.
        
        Args:
            spatial_velocity: Three-dimensional spatial velocity vector v^i in units
                where the speed of light c is explicit. Array must have exactly
                three components corresponding to spatial directions.
                
        Returns:
            Four-dimensional properly normalized four-velocity u^μ as numpy array
            with components [u^0, u^x, u^y, u^z]. The temporal component u^0 is
            positive (future-pointing timelike vector).
            
        Raises:
            ValueError: If spatial_velocity is not 3-dimensional or if the spatial
                velocity magnitude |v| ≥ c (superluminal motion).
                
        Examples:
            >>> u_field = FourVelocityField()
            >>> 
            >>> # Rest frame (zero spatial velocity)
            >>> v_rest = np.array([0, 0, 0])
            >>> u_rest = u_field.normalize(v_rest)  # Returns [c, 0, 0, 0]
            >>> 
            >>> # Motion in x-direction at half light speed
            >>> v_half_c = np.array([0.5, 0, 0])  # In units where c=1
            >>> u_moving = u_field.normalize(v_half_c)
            >>> # Returns [γc, γv, 0, 0] with γ = 1/√(1-0.25) ≈ 1.155
        """
        if len(spatial_velocity) != 3:
            raise ValueError("Spatial velocity must be 3-dimensional")
        
        c = PhysicalConstants.c
        v_squared = np.dot(spatial_velocity, spatial_velocity)
        
        if v_squared >= c**2:
            raise ValueError("Spatial velocity exceeds speed of light")
        
        # Lorentz factor
        gamma = 1.0 / np.sqrt(1 - v_squared / c**2)
        
        # Four-velocity
        u_mu = np.zeros(4)
        u_mu[0] = gamma * c  # u^0 = γc
        u_mu[1:] = gamma * spatial_velocity  # u^i = γv^i
        
        return u_mu


class ShearStressField(Field):
    """Shear stress π^{μν} (symmetric, traceless tensor)"""
    
    def __init__(self, metric: Optional[Metric] = None):
        properties = FieldProperties(
            name='pi',
            latex_symbol='\\pi',
            indices=['mu', 'nu'],
            index_types=['contravariant', 'contravariant'],
            engineering_dimension=2.0,  # [π] = M L^{-1} T^{-2} ~ M^2
            canonical_dimension=2.0,
            is_symmetric=True,
            is_traceless=True,
            is_spatial=True,  # Orthogonal to 4-velocity
            constraints=['pi^mu_mu = 0', 'pi^mu_nu = pi^nu_mu', 'u_mu * pi^mu_nu = 0']
        )
        super().__init__(properties, metric)
    
    def evolution_equation(self, tau_pi: float = 1.0, eta: float = 1.0, 
                          **kwargs) -> sp.Expr:
        """
        Israel-Stewart relaxation equation for shear stress tensor evolution.
        
        Returns the symbolic form of the evolution equation governing the shear
        stress tensor π^{μν} in Israel-Stewart theory. This second-order hyperbolic
        equation ensures causal propagation of viscous stresses with finite
        relaxation time τ_π.
        
        Evolution Equation:
            τ_π ∂_t π^{μν} + π^{μν} = 2η σ^{μν} - τ_π π^{μν} θ + O(higher order)
            
            where:
            - σ^{μν}: Shear tensor (symmetric, traceless, spatial)
            - θ = ∇_μ u^μ: Expansion scalar
            - η: Shear viscosity coefficient
            - τ_π: Shear relaxation time
            
        Args:
            tau_pi: Shear relaxation time parameter [time units]
            eta: Shear viscosity coefficient [M L^{-1} T^{-1}]
            **kwargs: Additional parameters for higher-order terms
            
        Returns:
            SymPy expression representing the evolution equation. Can be used for
            symbolic manipulation, linearization, or numerical implementation.
        """
        t = sp.Symbol('t')
        theta = sp.Symbol('theta')  # Expansion scalar
        sigma = sp.Function('sigma')  # Shear tensor
        
        # τ_π ∂_t π^{μν} + π^{μν} = 2η σ^{μν} - τ_π π^{μν} θ + ...
        evolution = (tau_pi * sp.Derivative(self.symbol, t) + 
                    self.symbol - 
                    2 * eta * sigma(t) + 
                    tau_pi * self.symbol * theta)
        
        return evolution


class BulkPressureField(Field):
    """Bulk pressure Π (scalar field)"""
    
    def __init__(self, metric: Optional[Metric] = None):
        properties = FieldProperties(
            name='Pi',
            latex_symbol='\\Pi',
            indices=[],
            index_types=[],
            engineering_dimension=2.0,  # Same as shear stress
            canonical_dimension=2.0
        )
        super().__init__(properties, metric)
    
    def evolution_equation(self, tau_Pi: float = 1.0, zeta: float = 1.0,
                          **kwargs) -> sp.Expr:
        """Israel-Stewart evolution equation for bulk pressure"""
        t = sp.Symbol('t')
        theta = sp.Symbol('theta')  # Expansion scalar
        
        # τ_Π ∂_t Π + Π = -ζ θ - τ_Π Π θ + ...
        evolution = (tau_Pi * sp.Derivative(self.symbol, t) + 
                    self.symbol + 
                    zeta * theta + 
                    tau_Pi * self.symbol * theta)
        
        return evolution


class HeatFluxField(Field):
    """Heat flux q^μ (vector field orthogonal to velocity)"""
    
    def __init__(self, metric: Optional[Metric] = None):
        properties = FieldProperties(
            name='q',
            latex_symbol='q',
            indices=['mu'],
            index_types=['contravariant'],
            engineering_dimension=3.0,  # [q] ~ M^3
            canonical_dimension=3.0,
            is_spatial=True,
            constraints=['u_mu * q^mu = 0']  # Orthogonal to velocity
        )
        super().__init__(properties, metric)
    
    def evolution_equation(self, tau_q: float = 1.0, kappa: float = 1.0,
                          **kwargs) -> sp.Expr:
        """Israel-Stewart evolution equation for heat flux"""
        t = sp.Symbol('t')
        theta = sp.Symbol('theta')
        T = sp.Symbol('T')  # Temperature
        alpha = 1 / T  # Inverse temperature
        
        # τ_q ∂_t q^μ + q^μ = -κ T ∇^μ α - τ_q q^μ θ + ...
        evolution = (tau_q * sp.Derivative(self.symbol, t) + 
                    self.symbol +
                    kappa * T * sp.Derivative(alpha, t) +
                    tau_q * self.symbol * theta)
        
        return evolution


class FieldRegistry:
    """Registry to manage all fields in the theory"""
    
    def __init__(self):
        """Initialize empty field registry"""
        self.fields: Dict[str, Field] = {}
        self.response_fields: Dict[str, ResponseField] = {}
    
    def register_field(self, field: Field) -> None:
        """Register a field and its response field"""
        self.fields[field.name] = field
        self.response_fields[f"tilde_{field.name}"] = field.response
    
    def create_is_fields(self, metric: Optional[Metric] = None) -> None:
        """Create all Israel-Stewart theory fields"""
        fields_to_create = [
            EnergyDensityField(metric),
            FourVelocityField(metric), 
            ShearStressField(metric),
            BulkPressureField(metric),
            HeatFluxField(metric)
        ]
        
        for field in fields_to_create:
            self.register_field(field)
    
    def get_field(self, name: str) -> Optional[Field]:
        """Get field by name"""
        return self.fields.get(name)
    
    def get_response_field(self, name: str) -> Optional[ResponseField]:
        """Get response field by name"""
        return self.response_fields.get(name)
    
    def list_fields(self) -> List[str]:
        """List all registered field names"""
        return list(self.fields.keys())
    
    def list_response_fields(self) -> List[str]:
        """List all response field names"""
        return list(self.response_fields.keys())
    
    def field_dimensions(self) -> Dict[str, float]:
        """Get engineering dimensions of all fields"""
        return {name: field.dimension for name, field in self.fields.items()}
    
    def __len__(self) -> int:
        """Number of registered fields"""
        return len(self.fields)
    
    def __contains__(self, name: str) -> bool:
        """Check if field is registered"""
        return name in self.fields or name in self.response_fields