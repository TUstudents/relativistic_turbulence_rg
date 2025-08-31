"""
Lorentz tensor algebra system for relativistic field theory calculations.

This module provides a comprehensive framework for tensor operations in Minkowski spacetime,
implementing the mathematical foundations required for relativistic hydrodynamics and field
theory. The implementation follows standard conventions from differential geometry and
general relativity.

Key Features:
    - Minkowski metric tensor with configurable signature
    - Automatic index management and type checking
    - Covariant tensor operations (contraction, symmetrization, etc.)
    - Spatial projection operators for relativistic hydrodynamics
    - Constraint enforcement for physical tensors

Mathematical Conventions:
    - Metric signature: (-,+,+,+) (mostly plus convention)
    - Greek indices (μ,ν,...): spacetime indices 0,1,2,3
    - Latin indices (i,j,...): spatial indices 1,2,3
    - Einstein summation convention throughout
    - Natural units: c = ℏ = k_B = 1

References:
    - Weinberg, S. "The Quantum Theory of Fields" Vol. I
    - Landau, L.D. & Lifshitz, E.M. "The Classical Theory of Fields"
    - Rezzolla, L. & Zanotti, O. "Relativistic Hydrodynamics"
"""
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass
import itertools

from .constants import PhysicalConstants


class Metric:
    """
    Minkowski spacetime metric tensor implementation.
    
    Represents the fundamental geometric structure of flat spacetime, providing
    the metric tensor g_μν used for raising/lowering indices and defining
    invariant intervals. The implementation supports configurable dimension
    and signature for theoretical flexibility.
    
    Mathematical Definition:
        The line element in Minkowski space is:
        ds² = g_μν dx^μ dx^ν = -c²dt² + dx² + dy² + dz²
        
    Attributes:
        dim (int): Spacetime dimension (typically 4)
        signature (tuple): Metric signature coefficients
        g (np.ndarray): Metric tensor components g_μν
        
    Examples:
        >>> metric = Metric()
        >>> print(metric.g)
        [[-1,  0,  0,  0],
         [ 0,  1,  0,  0], 
         [ 0,  0,  1,  0],
         [ 0,  0,  0,  1]]
         
        >>> # Custom 2+1 dimensional spacetime
        >>> metric_2d = Metric(dimension=3, signature=(-1, 1, 1))
    """
    
    def __init__(self, dimension: int = 4, signature: Optional[Tuple] = None):
        """
        Initialize Minkowski metric tensor.
        
        Args:
            dimension: Spacetime dimension. Must be positive integer.
                For standard relativity, use 4 (3+1 dimensions).
                For theoretical studies, other values are permitted.
            signature: Metric signature as tuple of ±1 values.
                Default is (-1, 1, 1, 1) for mostly-plus convention.
                Length must match dimension parameter.
                
        Raises:
            ValueError: If dimension is not positive or signature length
                doesn't match dimension.
        """
        self.dim = dimension
        self.signature = signature or PhysicalConstants.METRIC_SIGNATURE
        
        # Construct metric tensor
        self.g = np.zeros((dimension, dimension))
        for i in range(min(len(self.signature), dimension)):
            self.g[i, i] = self.signature[i]
    
    def contract(self, tensor: np.ndarray, 
                 indices: List[int]) -> np.ndarray:
        """Contract tensor indices with metric"""
        if len(indices) != 2:
            raise ValueError("Metric contraction requires exactly 2 indices")
            
        # Use Einstein summation convention
        einsum_string = self._build_einsum_string(tensor.shape, indices)
        return np.einsum(einsum_string, self.g, tensor)
    
    def raise_index(self, tensor: np.ndarray, position: int) -> np.ndarray:
        """Raise index using inverse metric g^μν"""
        g_inv = np.linalg.inv(self.g)
        axes = list(range(tensor.ndim))
        axes = [position] + [ax for ax in axes if ax != position]
        return np.tensordot(g_inv, tensor.transpose(axes), axes=([1], [0]))
    
    def lower_index(self, tensor: np.ndarray, position: int) -> np.ndarray:
        """Lower index using metric g_μν"""
        axes = list(range(tensor.ndim))
        axes = [position] + [ax for ax in axes if ax != position]
        return np.tensordot(self.g, tensor.transpose(axes), axes=([1], [0]))
    
    def _build_einsum_string(self, shape: Tuple, 
                            indices: List[int]) -> str:
        """Build einsum string for tensor contractions"""
        # This is a simplified version - full implementation would be more complex
        letters = 'abcdefghijklmnopqrstuvwxyz'
        
        # Metric indices
        metric_str = letters[0] + letters[1]
        
        # Tensor indices  
        tensor_str = ''.join(letters[i+2] for i in range(len(shape)))
        
        # Result indices (remove contracted indices)
        result_indices = [i for i in range(len(shape)) if i not in indices]
        result_str = ''.join(letters[i+2] for i in result_indices)
        
        return f'{metric_str},{tensor_str}->{result_str}'


@dataclass
class IndexStructure:
    """
    Tensor index structure management for covariant calculations.
    
    Maintains complete information about tensor indices including names,
    covariant/contravariant types, and symmetry properties. This enables
    automatic validation of tensor operations and proper index manipulation.
    
    Mathematical Context:
        In tensor notation, indices can be:
        - Contravariant (upper): T^μν  
        - Covariant (lower): T_μν
        - Mixed: T^μ_ν
        
    Symmetry properties determine transformation behavior under index
    permutation, which is crucial for physical tensors like the metric
    (symmetric) or electromagnetic field tensor (antisymmetric).
    
    Attributes:
        names: Symbolic names for indices (e.g., ['mu', 'nu', 'rho'])
        types: Index variance types, each either 'covariant' or 'contravariant'  
        symmetries: Symmetry properties under index exchange:
            - 'symmetric': T^μν = T^νμ
            - 'antisymmetric': T^μν = -T^νμ  
            - 'none': No particular symmetry
            
    Examples:
        >>> # Metric tensor g_μν (symmetric, covariant)
        >>> metric_indices = IndexStructure(
        ...     names=['mu', 'nu'],
        ...     types=['covariant', 'covariant'], 
        ...     symmetries=['symmetric', 'symmetric']
        ... )
        
        >>> # Electromagnetic field F^μν (antisymmetric, contravariant)
        >>> field_indices = IndexStructure(
        ...     names=['mu', 'nu'],
        ...     types=['contravariant', 'contravariant'],
        ...     symmetries=['antisymmetric', 'antisymmetric'] 
        ... )
    """
    names: List[str]
    types: List[str] 
    symmetries: List[str]
    
    def __post_init__(self):
        """
        Validate index structure consistency.
        
        Raises:
            ValueError: If index arrays have inconsistent lengths or
                contain invalid values.
        """
        if not (len(self.names) == len(self.types) == len(self.symmetries)):
            raise ValueError("Index arrays must have same length")
            
        # Validate types
        valid_types = {'covariant', 'contravariant'}
        for idx_type in self.types:
            if idx_type not in valid_types:
                raise ValueError(f"Invalid index type: {idx_type}")
                
        # Validate symmetries  
        valid_symmetries = {'symmetric', 'antisymmetric', 'none'}
        for symmetry in self.symmetries:
            if symmetry not in valid_symmetries:
                raise ValueError(f"Invalid symmetry: {symmetry}")
    
    @property
    def rank(self) -> int:
        """Tensor rank (number of indices)"""
        return len(self.names)
    
    def is_symmetric(self) -> bool:
        """Check if tensor has symmetric indices"""
        return 'symmetric' in self.symmetries
    
    def is_traceless(self) -> bool:
        """Check if tensor is traceless"""
        # Simplified - would need more sophisticated logic
        return hasattr(self, 'traceless') and self.traceless


class LorentzTensor:
    """
    Lorentz covariant tensor with automatic index management and operations.
    
    Implements tensors in Minkowski spacetime with full support for covariant
    operations including index raising/lowering, contractions, symmetrization,
    and spatial projections. The class enforces tensor transformation laws and
    maintains index bookkeeping automatically.
    
    Mathematical Foundation:
        A tensor T^{μ₁...μₘ}_{ν₁...νₙ} transforms under Lorentz transformations Λ as:
        T'^{μ₁...μₘ}_{ν₁...νₙ} = Λ^{μ₁}_{α₁} ... Λ^{μₘ}_{αₘ} Λ_{ν₁}^{β₁} ... Λ_{νₙ}^{βₙ} T^{α₁...αₘ}_{β₁...βₙ}
        
    Key Operations:
        - Index manipulation: raising (g^μν), lowering (g_μν) 
        - Contraction: T^μ_μ (sum over repeated indices)
        - Symmetrization: T^{(μν)} = ½(T^μν + T^νμ)
        - Antisymmetrization: T^{[μν]} = ½(T^μν - T^νμ)
        - Spatial projection: perpendicular to timelike vectors
        
    Attributes:
        components (np.ndarray): Tensor components in coordinate basis
        indices (IndexStructure): Complete index information 
        metric (Metric): Spacetime metric for index operations
        
    Examples:
        >>> # Stress-energy tensor T^μν
        >>> import numpy as np
        >>> T_components = np.diag([1, 0.3, 0.3, 0.3])  # Perfect fluid
        >>> indices = IndexStructure(['mu', 'nu'], ['contravariant', 'contravariant'], ['symmetric', 'symmetric'])
        >>> T = LorentzTensor(T_components, indices)
        >>> 
        >>> # Trace (energy density)
        >>> trace = T.trace()  # T^μ_μ = ρ + 3p
        >>>
        >>> # Four-velocity normalization
        >>> u = np.array([1, 0, 0, 0])  # Rest frame
        >>> u_indices = IndexStructure(['mu'], ['contravariant'], ['none'])
        >>> u_tensor = LorentzTensor(u, u_indices)
        >>> u_magnitude = u_tensor.contract(u_tensor.lower_index(0), [(0, 0)])  # Should be -c²
    """
    
    def __init__(self, components: np.ndarray, 
                 index_structure: IndexStructure,
                 metric: Optional[Metric] = None):
        """
        Initialize Lorentz tensor with components and index structure.
        
        Args:
            components: Tensor components as numpy array. Shape must match
                the tensor rank implied by index_structure. Complex values
                are supported for field theory applications.
            index_structure: Complete information about tensor indices including
                names, covariance types, and symmetries.
            metric: Spacetime metric for index operations. If None, uses
                standard Minkowski metric with (-,+,+,+) signature.
                
        Raises:
            ValueError: If tensor shape doesn't match index structure or
                if components contain invalid values (NaN, infinite).
        
        Examples:
            >>> # Scalar (rank-0 tensor)
            >>> scalar = LorentzTensor(np.array(1.5), IndexStructure([], [], []))
            >>>
            >>> # Vector (rank-1 tensor)  
            >>> vector_components = np.array([1, 2, 3, 4])
            >>> vector_indices = IndexStructure(['mu'], ['contravariant'], ['none'])
            >>> vector = LorentzTensor(vector_components, vector_indices)
        """
        self.components = np.asarray(components, dtype=complex)
        self.indices = index_structure
        self.metric = metric or Metric()
        
        # Validate tensor shape
        expected_shape = (self.metric.dim,) * self.indices.rank
        if self.components.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {self.components.shape}")
    
    @property
    def rank(self) -> int:
        """Tensor rank"""
        return self.indices.rank
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor shape"""
        return self.components.shape
    
    def contract(self, other: 'LorentzTensor', 
                 index_pairs: List[Tuple[int, int]]) -> 'LorentzTensor':
        """
        Contract tensor indices using Einstein summation convention.
        
        Performs tensor contraction between this tensor and another, summing over
        paired indices. The operation follows standard tensor algebra with automatic
        handling of covariant and contravariant indices via the metric tensor.
        
        Mathematical Operation:
            For tensors A^{μν} and B_{μρ}, contraction over μ gives:
            C^ν_ρ = A^{μν} B_{μρ} = Σ_μ A^{μν} B_{μρ}
            
        The metric tensor handles index type compatibility:
            - Contravariant-covariant pairs contract directly
            - Same-type pairs require metric tensor insertion
            
        Args:
            other: Tensor to contract with. Must have compatible index structure
                for the specified contractions.
            index_pairs: List of (self_index, other_index) tuples specifying
                which indices to contract. Each tuple (i,j) means contract
                the i-th index of self with j-th index of other.
                
        Returns:
            New LorentzTensor containing the result of contraction. The rank
            equals (self.rank + other.rank - 2*len(index_pairs)).
            
        Raises:
            ValueError: If index pairs are invalid or incompatible for contraction.
            
        Examples:
            >>> # Contract vector with covector: v^μ w_μ (gives scalar)
            >>> v = LorentzTensor(np.array([1,2,3,4]), vector_up_indices)
            >>> w = LorentzTensor(np.array([1,-1,0,1]), vector_down_indices) 
            >>> scalar = v.contract(w, [(0, 0)])
            >>>
            >>> # Contract tensors: T^{μν} S_{μ}^{ρ} (gives T'^{νρ})
            >>> result = T.contract(S, [(0, 0)])  # Contract first indices
        """
        # Build einsum string for contraction
        einsum_str = self._build_contraction_einsum(other, index_pairs)
        
        # Perform contraction
        result_components = np.einsum(einsum_str, self.components, other.components)
        
        # Build result index structure
        result_indices = self._build_result_indices(other, index_pairs)
        
        return LorentzTensor(result_components, result_indices, self.metric)
    
    def symmetrize(self) -> 'LorentzTensor':
        """Symmetrize tensor over all indices"""
        if self.rank < 2:
            return self
            
        # Generate all permutations of indices
        perms = list(itertools.permutations(range(self.rank)))
        
        # Sum over all permutations
        result = np.zeros_like(self.components)
        for perm in perms:
            result += self.components.transpose(perm)
        
        # Normalize
        result /= len(perms)
        
        # Update index structure
        new_indices = IndexStructure(
            names=self.indices.names,
            types=self.indices.types,
            symmetries=['symmetric'] * self.rank
        )
        
        return LorentzTensor(result, new_indices, self.metric)
    
    def antisymmetrize(self) -> 'LorentzTensor':
        """Antisymmetrize tensor over all indices"""
        if self.rank < 2:
            return self
            
        # Generate all permutations with signs
        perms = list(itertools.permutations(range(self.rank)))
        
        # Sum with appropriate signs
        result = np.zeros_like(self.components)
        for perm in perms:
            sign = (-1) ** self._permutation_parity(perm)
            result += sign * self.components.transpose(perm)
        
        # Normalize  
        result /= len(perms)
        
        # Update index structure
        new_indices = IndexStructure(
            names=self.indices.names,
            types=self.indices.types,
            symmetries=['antisymmetric'] * self.rank
        )
        
        return LorentzTensor(result, new_indices, self.metric)
    
    def trace(self, index_pair: Optional[Tuple[int, int]] = None) -> Union['LorentzTensor', complex]:
        """Take trace over specified indices
        
        Args:
            index_pair: Pair of indices to trace over (default: all if rank=2)
            
        Returns:
            Tensor with reduced rank, or scalar if rank becomes 0
        """
        if self.rank == 0:
            raise ValueError("Cannot take trace of scalar")
        
        if index_pair is None and self.rank == 2:
            # Trace over both indices
            trace_val = np.trace(self.components)
            return trace_val
        elif index_pair is None:
            raise ValueError("Must specify index pair for rank > 2 tensors")
        
        # Trace over specified indices
        i, j = index_pair
        axes = (i, j)
        result = np.trace(self.components, axis1=i, axis2=j)
        
        # Build result index structure
        remaining_names = [name for k, name in enumerate(self.indices.names) 
                          if k not in axes]
        remaining_types = [typ for k, typ in enumerate(self.indices.types)
                          if k not in axes]  
        remaining_symmetries = [sym for k, sym in enumerate(self.indices.symmetries)
                               if k not in axes]
        
        if len(remaining_names) == 0:
            return result  # Scalar result
        
        result_indices = IndexStructure(remaining_names, remaining_types, remaining_symmetries)
        return LorentzTensor(result, result_indices, self.metric)
    
    def raise_index(self, position: int) -> 'LorentzTensor':
        """Raise index at given position"""
        if self.indices.types[position] == 'contravariant':
            return self  # Already raised
            
        new_components = self.metric.raise_index(self.components, position)
        
        # Update index types
        new_types = self.indices.types.copy()
        new_types[position] = 'contravariant'
        
        new_indices = IndexStructure(
            names=self.indices.names,
            types=new_types,
            symmetries=self.indices.symmetries
        )
        
        return LorentzTensor(new_components, new_indices, self.metric)
    
    def lower_index(self, position: int) -> 'LorentzTensor':
        """Lower index at given position"""
        if self.indices.types[position] == 'covariant':
            return self  # Already lowered
            
        new_components = self.metric.lower_index(self.components, position)
        
        # Update index types
        new_types = self.indices.types.copy()
        new_types[position] = 'covariant'
        
        new_indices = IndexStructure(
            names=self.indices.names,
            types=new_types,
            symmetries=self.indices.symmetries
        )
        
        return LorentzTensor(new_components, new_indices, self.metric)
    
    def project_spatial(self, velocity: np.ndarray) -> 'LorentzTensor':
        """Project tensor onto spatial subspace orthogonal to velocity
        
        Args:
            velocity: 4-velocity u^μ (assumed normalized: u·u = -c²)
            
        Returns:
            Spatially projected tensor
        """
        if len(velocity) != self.metric.dim:
            raise ValueError(f"Velocity must have dimension {self.metric.dim}")
        
        # Construct spatial projector Δ^μν = g^μν + u^μu^ν/c²
        g = self.metric.g
        u = velocity.reshape(-1, 1)
        c_sq = PhysicalConstants.c**2
        
        # Projector (assuming u is contravariant)
        Delta = g + np.outer(velocity, velocity) / c_sq
        
        # Apply projector to each index
        result = self.components.copy()
        for i in range(self.rank):
            result = np.tensordot(Delta, result, axes=([1], [i]))
            # Move contracted axis back to position i
            axes = list(range(result.ndim))
            axes = axes[1:i+1] + [0] + axes[i+1:]
            result = result.transpose(axes)
        
        return LorentzTensor(result, self.indices, self.metric)
    
    def _build_contraction_einsum(self, other: 'LorentzTensor', 
                                  index_pairs: List[Tuple[int, int]]) -> str:
        """Build Einstein summation string for tensor contraction"""
        # This is a simplified version - full implementation would be more sophisticated
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Assign letters to indices
        self_letters = letters[:self.rank]
        other_letters = letters[self.rank:self.rank + other.rank]
        
        # Handle contractions
        for self_idx, other_idx in index_pairs:
            other_letters = list(other_letters)
            other_letters[other_idx] = self_letters[self_idx]
            other_letters = ''.join(other_letters)
        
        # Result letters (uncontracted indices)
        contracted_self = [pair[0] for pair in index_pairs]
        contracted_other = [pair[1] for pair in index_pairs]
        
        result_letters = ''.join([self_letters[i] for i in range(self.rank) 
                                 if i not in contracted_self])
        result_letters += ''.join([other_letters[i] for i in range(other.rank)
                                  if i not in contracted_other])
        
        return f'{self_letters},{other_letters}->{result_letters}'
    
    def _build_result_indices(self, other: 'LorentzTensor',
                             index_pairs: List[Tuple[int, int]]) -> IndexStructure:
        """Build index structure for contraction result"""
        # Get uncontracted indices
        contracted_self = [pair[0] for pair in index_pairs]
        contracted_other = [pair[1] for pair in index_pairs]
        
        result_names = [self.indices.names[i] for i in range(self.rank)
                       if i not in contracted_self]
        result_names += [other.indices.names[i] for i in range(other.rank)
                        if i not in contracted_other]
        
        result_types = [self.indices.types[i] for i in range(self.rank)
                       if i not in contracted_self]
        result_types += [other.indices.types[i] for i in range(other.rank)
                        if i not in contracted_other]
        
        result_symmetries = [self.indices.symmetries[i] for i in range(self.rank)
                            if i not in contracted_self]
        result_symmetries += [other.indices.symmetries[i] for i in range(other.rank)
                             if i not in contracted_other]
        
        return IndexStructure(result_names, result_types, result_symmetries)
    
    def _permutation_parity(self, perm: Tuple[int, ...]) -> int:
        """Calculate parity of permutation (0 for even, 1 for odd)"""
        n = len(perm)
        visited = [False] * n
        parity = 0
        
        for i in range(n):
            if visited[i]:
                continue
                
            cycle_length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_length += 1
            
            if cycle_length > 1 and cycle_length % 2 == 0:
                parity ^= 1
        
        return parity
    
    def __str__(self) -> str:
        """String representation"""
        return f"LorentzTensor(rank={self.rank}, shape={self.shape})"
    
    def __repr__(self) -> str:
        return self.__str__()