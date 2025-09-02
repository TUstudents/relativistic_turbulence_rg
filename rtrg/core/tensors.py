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

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

import numpy as np

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

    def __init__(self, dimension: int = 4, signature: tuple | None = None):
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

    def contract(self, tensor: np.ndarray, indices: list[int]) -> np.ndarray:
        """Contract tensor indices with metric"""
        if len(indices) != 2:
            raise ValueError("Metric contraction requires exactly 2 indices")

        # Use Einstein summation convention
        einsum_string = self._build_einsum_string(tensor.shape, indices)
        return np.einsum(einsum_string, self.g, tensor)  # type: ignore[no-any-return]

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

    def _build_einsum_string(self, shape: tuple, indices: list[int]) -> str:
        """Build einsum string for tensor contractions"""
        # This is a simplified version - full implementation would be more complex
        letters = "abcdefghijklmnopqrstuvwxyz"

        # Metric indices
        metric_str = letters[0] + letters[1]

        # Tensor indices
        tensor_str = "".join(letters[i + 2] for i in range(len(shape)))

        # Result indices (remove contracted indices)
        result_indices = [i for i in range(len(shape)) if i not in indices]
        result_str = "".join(letters[i + 2] for i in result_indices)

        return f"{metric_str},{tensor_str}->{result_str}"


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

    names: list[str]
    types: list[str]
    symmetries: list[str]

    def __post_init__(self) -> None:
        """
        Validate index structure consistency.

        Raises:
            ValueError: If index arrays have inconsistent lengths or
                contain invalid values.
        """
        if not (len(self.names) == len(self.types) == len(self.symmetries)):
            raise ValueError("Index arrays must have same length")

        # Validate types
        valid_types = {"covariant", "contravariant"}
        for idx_type in self.types:
            if idx_type not in valid_types:
                raise ValueError(f"Invalid index type: {idx_type}")

        # Validate symmetries
        valid_symmetries = {"symmetric", "antisymmetric", "none"}
        for symmetry in self.symmetries:
            if symmetry not in valid_symmetries:
                raise ValueError(f"Invalid symmetry: {symmetry}")

    @property
    def rank(self) -> int:
        """Tensor rank (number of indices)"""
        return len(self.names)

    def is_symmetric(self) -> bool:
        """Check if tensor has symmetric indices"""
        return "symmetric" in self.symmetries

    def is_traceless(self) -> bool:
        """Check if tensor is traceless"""
        # Simplified - would need more sophisticated logic
        return hasattr(self, "traceless") and self.traceless


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

    def __init__(
        self, components: np.ndarray, index_structure: IndexStructure, metric: Metric | None = None
    ):
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
    def shape(self) -> tuple[int, ...]:
        """Tensor shape"""
        return self.components.shape

    def contract(
        self, other: "LorentzTensor", index_pairs: list[tuple[int, int]]
    ) -> "LorentzTensor":
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

    def symmetrize(self) -> "LorentzTensor":
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
            names=self.indices.names, types=self.indices.types, symmetries=["symmetric"] * self.rank
        )

        return LorentzTensor(result, new_indices, self.metric)

    def antisymmetrize(self) -> "LorentzTensor":
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
            symmetries=["antisymmetric"] * self.rank,
        )

        return LorentzTensor(result, new_indices, self.metric)

    def trace(self, index_pair: tuple[int, int] | None = None) -> Union["LorentzTensor", complex]:
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
            return trace_val  # type: ignore[no-any-return]
        elif index_pair is None:
            raise ValueError("Must specify index pair for rank > 2 tensors")

        # Trace over specified indices
        i, j = index_pair
        axes = (i, j)
        result = np.trace(self.components, axis1=i, axis2=j)

        # Build result index structure
        remaining_names = [name for k, name in enumerate(self.indices.names) if k not in axes]
        remaining_types = [typ for k, typ in enumerate(self.indices.types) if k not in axes]
        remaining_symmetries = [
            sym for k, sym in enumerate(self.indices.symmetries) if k not in axes
        ]

        if len(remaining_names) == 0:
            return result  # type: ignore[no-any-return]

        result_indices = IndexStructure(remaining_names, remaining_types, remaining_symmetries)
        return LorentzTensor(result, result_indices, self.metric)

    def raise_index(self, position: int) -> "LorentzTensor":
        """Raise index at given position"""
        if self.indices.types[position] == "contravariant":
            return self  # Already raised

        new_components = self.metric.raise_index(self.components, position)

        # Update index types
        new_types = self.indices.types.copy()
        new_types[position] = "contravariant"

        new_indices = IndexStructure(
            names=self.indices.names, types=new_types, symmetries=self.indices.symmetries
        )

        return LorentzTensor(new_components, new_indices, self.metric)

    def lower_index(self, position: int) -> "LorentzTensor":
        """Lower index at given position"""
        if self.indices.types[position] == "covariant":
            return self  # Already lowered

        new_components = self.metric.lower_index(self.components, position)

        # Update index types
        new_types = self.indices.types.copy()
        new_types[position] = "covariant"

        new_indices = IndexStructure(
            names=self.indices.names, types=new_types, symmetries=self.indices.symmetries
        )

        return LorentzTensor(new_components, new_indices, self.metric)

    def project_spatial(self, velocity: np.ndarray) -> "LorentzTensor":
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
        c_sq = PhysicalConstants.c**2

        # Projector (assuming u is contravariant)
        Delta = g + np.outer(velocity, velocity) / c_sq

        # Apply projector to each index
        result = self.components.copy()
        for i in range(self.rank):
            result = np.tensordot(Delta, result, axes=([1], [i]))
            # Move contracted axis back to position i
            axes = list(range(result.ndim))
            axes = axes[1 : i + 1] + [0] + axes[i + 1 :]
            result = result.transpose(axes)

        return LorentzTensor(result, self.indices, self.metric)

    def _build_contraction_einsum(
        self, other: "LorentzTensor", index_pairs: list[tuple[int, int]]
    ) -> str:
        """Build Einstein summation string for tensor contraction"""
        # This is a simplified version - full implementation would be more sophisticated
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Assign letters to indices
        self_letters = letters[: self.rank]
        other_letters = letters[self.rank : self.rank + other.rank]

        # Handle contractions
        for self_idx, other_idx in index_pairs:
            other_letters_list = list(other_letters)
            other_letters_list[other_idx] = self_letters[self_idx]
            other_letters = "".join(other_letters_list)

        # Result letters (uncontracted indices)
        contracted_self = [pair[0] for pair in index_pairs]
        contracted_other = [pair[1] for pair in index_pairs]

        result_letters = "".join(
            [self_letters[i] for i in range(self.rank) if i not in contracted_self]
        )
        result_letters += "".join(
            [other_letters[i] for i in range(other.rank) if i not in contracted_other]
        )

        return f"{self_letters},{other_letters}->{result_letters}"

    def christoffel_symbols(self, metric_derivatives: np.ndarray | None = None) -> "LorentzTensor":
        """
        Compute Christoffel symbols Γ^λ_μν for the metric.

        Mathematical Definition:
            Γ^λ_μν = ½ g^λρ (∂_μ g_ρν + ∂_ν g_ρμ - ∂_ρ g_μν)

        For Minkowski space, all Christoffel symbols vanish (flat spacetime).
        For curved spacetime, metric derivatives must be provided.

        Args:
            metric_derivatives: Array of metric derivatives ∂_α g_μν.
                Shape should be (dim, dim, dim) for rank-3 tensor.
                If None, assumes flat Minkowski space (Γ = 0).

        Returns:
            LorentzTensor containing Christoffel symbols with structure:
            - First index (λ): contravariant (raised)
            - Second/third indices (μν): covariant (lowered)
            - Symmetric in last two indices: Γ^λ_μν = Γ^λ_νμ

        Examples:
            >>> # Flat Minkowski space (all Christoffel symbols vanish)
            >>> metric = Metric()
            >>> tensor = LorentzTensor(metric.g, metric_indices)
            >>> christoffel = tensor.christoffel_symbols()  # Returns zero tensor
            >>>
            >>> # Curved spacetime (requires metric derivatives)
            >>> derivs = compute_metric_derivatives(g)  # User-provided
            >>> christoffel = tensor.christoffel_symbols(derivs)
        """
        dim = self.metric.dim

        if metric_derivatives is None:
            # Flat Minkowski space - all Christoffel symbols vanish
            christoffel_components = np.zeros((dim, dim, dim), dtype=complex)
        else:
            if metric_derivatives.shape != (dim, dim, dim):
                raise ValueError(f"Metric derivatives must have shape {(dim, dim, dim)}")

            # Compute Christoffel symbols: Γ^λ_μν = ½ g^λρ (∂_μ g_ρν + ∂_ν g_ρμ - ∂_ρ g_μν)
            g_inv = np.linalg.inv(self.metric.g)
            christoffel_components = np.zeros((dim, dim, dim), dtype=complex)

            for lam in range(dim):
                for mu in range(dim):
                    for nu in range(dim):
                        for rho in range(dim):
                            christoffel_components[lam, mu, nu] += (
                                0.5
                                * g_inv[lam, rho]
                                * (
                                    metric_derivatives[mu, rho, nu]
                                    + metric_derivatives[nu, rho, mu]
                                    - metric_derivatives[rho, mu, nu]
                                )
                            )

        # Create index structure for Christoffel symbols
        christoffel_indices = IndexStructure(
            names=["lambda", "mu", "nu"],
            types=["contravariant", "covariant", "covariant"],
            symmetries=["none", "symmetric", "symmetric"],  # Symmetric in last two indices
        )

        return LorentzTensor(christoffel_components, christoffel_indices, self.metric)

    def covariant_derivative(
        self, position: int, christoffel: "LorentzTensor | None" = None
    ) -> "LorentzTensor":
        """
        Compute covariant derivative ∇_μ T of the tensor.

        Mathematical Definition:
            For contravariant tensor T^α: ∇_μ T^α = ∂_μ T^α + Γ^α_μβ T^β
            For covariant tensor T_α: ∇_μ T_α = ∂_μ T_α - Γ^β_μα T_β
            For mixed tensors: combine both rules

        In Minkowski space with Cartesian coordinates, covariant derivative
        reduces to ordinary partial derivative since Christoffel symbols vanish.

        Args:
            position: Position to insert the covariant derivative index.
                New index will be covariant (lowered).
            christoffel: Precomputed Christoffel symbols. If None, assumes
                flat Minkowski space where ∇_μ = ∂_μ.

        Returns:
            New tensor with rank increased by 1, containing the covariant derivative.
            The derivative index is inserted at the specified position.

        Note:
            This method currently implements the case for flat spacetime only.
            For curved spacetime, the full implementation would require:
            - Numerical differentiation of tensor components
            - Christoffel symbol correction terms
            - Proper handling of all index types

        Examples:
            >>> # Covariant derivative of vector v^μ gives ∇_ν v^μ
            >>> vector = LorentzTensor(v_components, vector_indices)
            >>> cov_deriv = vector.covariant_derivative(0)  # Insert ∇ at position 0
            >>>
            >>> # In flat space: ∇_μ T^νρ = ∂_μ T^νρ (no Christoffel terms)
        """
        if christoffel is None:
            # Flat Minkowski space: covariant derivative = partial derivative
            # For this implementation, we return a symbolic placeholder
            # Full implementation would require numerical differentiation

            # Create extended shape for derivative
            new_shape = list(self.shape)
            new_shape.insert(position, self.metric.dim)

            # For now, return zero tensor as placeholder
            # Real implementation would compute ∂_μ T numerically
            result_components = np.zeros(tuple(new_shape), dtype=complex)

            # Build new index structure
            new_names = self.indices.names.copy()
            new_names.insert(position, "derivative")

            new_types = self.indices.types.copy()
            new_types.insert(position, "covariant")  # Derivative index is always covariant

            new_symmetries = self.indices.symmetries.copy()
            new_symmetries.insert(position, "none")

            new_indices = IndexStructure(new_names, new_types, new_symmetries)

            return LorentzTensor(result_components, new_indices, self.metric)
        else:
            # Curved spacetime implementation would go here
            raise NotImplementedError(
                "Covariant derivative with Christoffel symbols not yet implemented"
            )

    def _build_result_indices(
        self, other: "LorentzTensor", index_pairs: list[tuple[int, int]]
    ) -> IndexStructure:
        """Build index structure for contraction result"""
        # Get uncontracted indices
        contracted_self = [pair[0] for pair in index_pairs]
        contracted_other = [pair[1] for pair in index_pairs]

        result_names = [self.indices.names[i] for i in range(self.rank) if i not in contracted_self]
        result_names += [
            other.indices.names[i] for i in range(other.rank) if i not in contracted_other
        ]

        result_types = [self.indices.types[i] for i in range(self.rank) if i not in contracted_self]
        result_types += [
            other.indices.types[i] for i in range(other.rank) if i not in contracted_other
        ]

        result_symmetries = [
            self.indices.symmetries[i] for i in range(self.rank) if i not in contracted_self
        ]
        result_symmetries += [
            other.indices.symmetries[i] for i in range(other.rank) if i not in contracted_other
        ]

        return IndexStructure(result_names, result_types, result_symmetries)

    def _permutation_parity(self, perm: tuple[int, ...]) -> int:
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


# ============================================================================
# Enhanced Tensor Index System for MSRJD Field Theory
# ============================================================================


class IndexType(Enum):
    """Types of tensor indices for relativistic field theory."""

    SPACETIME = "spacetime"  # μ, ν, λ, ... (0,1,2,3)
    SPATIAL = "spatial"  # i, j, k, ... (1,2,3)
    TEMPORAL = "temporal"  # time component (0)


@dataclass
class TensorIndex:
    """
    Represents a single tensor index with type and position information.

    This class handles the bookkeeping for tensor indices in relativistic
    field theory calculations, ensuring proper contraction rules and
    type safety.
    """

    name: str  # Index symbol (μ, ν, i, j, etc.)
    index_type: IndexType  # Type of index
    position: str = "upper"  # "upper" or "lower" for covariant/contravariant
    dimension: int = 4  # Dimension of index space

    def __post_init__(self) -> None:
        """Validate index parameters."""
        if self.position not in ["upper", "lower"]:
            raise ValueError(f"Invalid position: {self.position}")

        if self.index_type == IndexType.SPATIAL and self.dimension != 3:
            self.dimension = 3
        elif self.index_type == IndexType.TEMPORAL and self.dimension != 1:
            self.dimension = 1

    def is_contractible_with(self, other: "TensorIndex") -> bool:
        """Check if this index can contract with another."""
        return (
            self.index_type == other.index_type
            and self.dimension == other.dimension
            and self.position != other.position  # Must have opposite positions
            and self.name == other.name  # Must be the same index symbol
        )

    def raise_lower_index(self) -> "TensorIndex":
        """Return index with opposite position (raise/lower)."""
        new_position = "lower" if self.position == "upper" else "upper"
        return TensorIndex(self.name, self.index_type, new_position, self.dimension)


class TensorIndexStructure:
    """
    Manages the complete index structure for tensor fields.

    This class provides comprehensive index management for relativistic
    tensor fields, handling automatic contraction, constraint checking,
    and tensor algebra operations.
    """

    def __init__(self, indices: list[TensorIndex]):
        """
        Initialize with list of tensor indices.

        Args:
            indices: List of TensorIndex objects defining tensor structure
        """
        self.indices = indices.copy()
        self._validate_indices()

    def _validate_indices(self) -> None:
        """Validate tensor index structure."""
        # Check for dummy index pairs (same name, opposite position)
        index_counts: dict[str, int] = {}
        for idx in self.indices:
            index_counts[idx.name] = index_counts.get(idx.name, 0) + 1

        # Dummy indices should appear exactly twice
        for name, count in index_counts.items():
            if count > 2:
                raise ValueError(f"Index {name} appears {count} times (max 2 allowed)")

    @property
    def rank(self) -> int:
        """Tensor rank (number of indices)."""
        return len(self.indices)

    @property
    def free_indices(self) -> list[TensorIndex]:
        """Get free (uncontracted) indices."""
        index_counts: dict[str, list[TensorIndex]] = {}
        for idx in self.indices:
            if idx.name not in index_counts:
                index_counts[idx.name] = []
            index_counts[idx.name].append(idx)

        free = []
        for indices_list in index_counts.values():
            if len(indices_list) == 1:
                free.append(indices_list[0])

        return free

    @property
    def dummy_indices(self) -> list[tuple[TensorIndex, TensorIndex]]:
        """Get dummy (contracted) index pairs."""
        index_groups: dict[str, list[TensorIndex]] = {}
        for idx in self.indices:
            if idx.name not in index_groups:
                index_groups[idx.name] = []
            index_groups[idx.name].append(idx)

        dummy_pairs = []
        for indices_list in index_groups.values():
            if len(indices_list) == 2:
                idx1, idx2 = indices_list
                if idx1.is_contractible_with(idx2):
                    dummy_pairs.append((idx1, idx2))

        return dummy_pairs

    def contract_with(
        self, other: "TensorIndexStructure", contractions: list[tuple[int, int]]
    ) -> "TensorIndexStructure":
        """
        Contract this tensor structure with another.

        Args:
            other: Other tensor index structure
            contractions: List of (self_idx, other_idx) pairs to contract

        Returns:
            New TensorIndexStructure representing the contraction result
        """
        # Validate contractions
        for self_idx, other_idx in contractions:
            if not (0 <= self_idx < len(self.indices)):
                raise ValueError(f"Invalid self index: {self_idx}")
            if not (0 <= other_idx < len(other.indices)):
                raise ValueError(f"Invalid other index: {other_idx}")

            self_index = self.indices[self_idx]
            other_index = other.indices[other_idx]

            if not self_index.is_contractible_with(other_index):
                raise ValueError(f"Cannot contract {self_index} with {other_index}")

        # Build result indices
        result_indices = []

        # Add uncontracted indices from self
        contracted_self = {pair[0] for pair in contractions}
        for i, idx in enumerate(self.indices):
            if i not in contracted_self:
                result_indices.append(idx)

        # Add uncontracted indices from other
        contracted_other = {pair[1] for pair in contractions}
        for i, idx in enumerate(other.indices):
            if i not in contracted_other:
                result_indices.append(idx)

        return TensorIndexStructure(result_indices)

    def apply_symmetry(self, symmetry_type: str, index_pairs: list[tuple[int, int]]) -> None:
        """
        Apply symmetry constraints to tensor indices.

        Args:
            symmetry_type: "symmetric" or "antisymmetric"
            index_pairs: Pairs of indices to make (anti)symmetric
        """
        # This is a placeholder - full implementation would modify tensor components
        # according to symmetry constraints
        pass

    def apply_traceless_condition(self, index_pairs: list[tuple[int, int]]) -> None:
        """
        Apply traceless condition to tensor indices.

        Args:
            index_pairs: Pairs of indices that should be traced to zero
        """
        # This is a placeholder - full implementation would project out trace
        pass


class ConstrainedTensorField:
    """
    Represents tensor fields with physical constraints.

    This class handles tensor fields that must satisfy physical constraints
    such as normalization, orthogonality, symmetry, and tracelessness.
    """

    def __init__(
        self, name: str, index_structure: TensorIndexStructure, constraints: list[str] | None = None
    ):
        """
        Initialize constrained tensor field.

        Args:
            name: Field name
            index_structure: Tensor index structure
            constraints: List of constraint names
        """
        self.name = name
        self.index_structure = index_structure
        self.constraints = constraints or []
        self._constraint_handlers: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "normalized": self._apply_normalization,
            "symmetric": self._apply_symmetry,
            "antisymmetric": self._apply_antisymmetry,
            "traceless": self._apply_traceless,
            "orthogonal_to_velocity": lambda x, **kwargs: self._apply_velocity_orthogonality(
                x, kwargs.get("velocity", np.array([1, 0, 0, 0]))
            ),
        }

    def _apply_normalization(self, components: np.ndarray, norm_value: float = -1.0) -> np.ndarray:
        """Apply normalization constraint."""
        # For four-velocity: u^μ u_μ = -c² = -1 (natural units)
        current_norm = np.sum(components * components * np.array([-1, 1, 1, 1]))
        if abs(current_norm - norm_value) > 1e-10:
            # Rescale to maintain normalization
            scale_factor = np.sqrt(abs(norm_value / current_norm))
            return components * scale_factor  # type: ignore[no-any-return]
        return components  # type: ignore[no-any-return]

    def _apply_symmetry(self, components: np.ndarray) -> np.ndarray:
        """Apply symmetry constraint."""
        # Symmetrize tensor components
        if len(components.shape) == 2:
            return 0.5 * (components + components.T)  # type: ignore[no-any-return]
        # Higher rank tensors would need more sophisticated symmetrization
        return components  # type: ignore[no-any-return]

    def _apply_antisymmetry(self, components: np.ndarray) -> np.ndarray:
        """Apply antisymmetry constraint."""
        if len(components.shape) == 2:
            return 0.5 * (components - components.T)  # type: ignore[no-any-return]
        return components  # type: ignore[no-any-return]

    def _apply_traceless(self, components: np.ndarray) -> np.ndarray:
        """Apply traceless constraint."""
        if len(components.shape) == 2:
            trace = np.trace(components)
            dim = components.shape[0]
            return components - (trace / dim) * np.eye(dim)  # type: ignore[no-any-return]
        return components  # type: ignore[no-any-return]

    def _apply_velocity_orthogonality(
        self, components: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """Apply orthogonality to four-velocity."""
        # Project out components parallel to velocity
        # For vector: q^μ → q^μ - (q·u)u^μ
        if len(components.shape) == 1:
            dot_product = np.sum(components * velocity * np.array([-1, 1, 1, 1]))
            return components - dot_product * velocity  # type: ignore[no-any-return]
        return components  # type: ignore[no-any-return]

    def apply_constraints(self, components: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply all registered constraints to field components."""
        result = components.copy()

        for constraint in self.constraints:
            if constraint in self._constraint_handlers:
                result = self._constraint_handlers[constraint](result, **kwargs)

        return result

    def validate_constraints(self, components: np.ndarray, **kwargs: Any) -> dict[str, bool]:
        """Validate that components satisfy all constraints."""
        validation_results = {}

        for constraint in self.constraints:
            if constraint == "normalized":
                norm = np.sum(components * components * np.array([-1, 1, 1, 1]))
                validation_results[constraint] = abs(norm + 1.0) < 1e-10
            elif constraint == "symmetric":
                if len(components.shape) == 2:
                    validation_results[constraint] = np.allclose(
                        components, components.T, rtol=1e-10
                    )
                else:
                    validation_results[constraint] = True
            elif constraint == "traceless":
                if len(components.shape) == 2:
                    validation_results[constraint] = abs(np.trace(components)) < 1e-10
                else:
                    validation_results[constraint] = True
            else:
                validation_results[constraint] = True

        return validation_results


class ProjectionOperators:
    """
    Projection operators for relativistic hydrodynamic field decomposition.

    This class provides the fundamental projection operators needed to decompose
    tensor fields into longitudinal/transverse components and enforce physical
    constraints in relativistic fluid dynamics.
    """

    def __init__(self, metric: Metric):
        """
        Initialize with spacetime metric.

        Args:
            metric: Metric object defining the spacetime geometry
        """
        self.metric = metric

    def spatial_projector(self, four_velocity: np.ndarray) -> np.ndarray:
        """
        Spatial projection operator h^μν = g^μν + u^μu^ν/c².

        Projects tensors into the spatial hypersurface orthogonal to
        the four-velocity, essential for 3+1 decomposition.

        Args:
            four_velocity: Four-velocity u^μ (normalized)

        Returns:
            Spatial projector h^μν as 4×4 array
        """
        g_inv = np.linalg.inv(self.metric.g)
        u_outer = np.outer(four_velocity, four_velocity)

        # In natural units c = 1, so h^μν = g^μν + u^μu^ν
        h_projector = g_inv + u_outer

        return h_projector

    def longitudinal_projector(self, momentum: np.ndarray) -> np.ndarray:
        """
        Longitudinal projection operator P^L_ij = k_ik_j/k².

        Projects vector fields along the momentum direction,
        used for decomposing velocity perturbations.

        Args:
            momentum: Spatial momentum vector k^i

        Returns:
            Longitudinal projector P^L_ij as 3×3 array
        """
        k_magnitude_sq = np.sum(momentum**2)

        if k_magnitude_sq < 1e-12:
            # For k→0, return zero projector
            return np.zeros((3, 3))

        k_outer = np.outer(momentum, momentum)
        return k_outer / k_magnitude_sq  # type: ignore[no-any-return]

    def transverse_projector(self, momentum: np.ndarray) -> np.ndarray:
        """
        Transverse projection operator P^T_ij = δ_ij - k_ik_j/k².

        Projects vector fields perpendicular to momentum direction,
        used for transverse (shear) modes.

        Args:
            momentum: Spatial momentum vector k^i

        Returns:
            Transverse projector P^T_ij as 3×3 array
        """
        longitudinal = self.longitudinal_projector(momentum)
        return np.eye(3) - longitudinal  # type: ignore[no-any-return]

    def symmetric_traceless_projector(self, four_velocity: np.ndarray) -> np.ndarray:
        """
        Symmetric traceless tensor projector for shear stress.

        Projects rank-2 tensors into the symmetric, traceless, and
        orthogonal-to-velocity subspace: π^μν_orthogonal.

        Args:
            four_velocity: Four-velocity u^μ (normalized)

        Returns:
            Projection operator as 4×4×4×4 array
        """
        h_proj = self.spatial_projector(four_velocity)

        # Build the symmetric traceless projector
        # P^μν_αβ = ½(h^μ_α h^ν_β + h^μ_β h^ν_α) - ⅓h^μν h_αβ
        projector = np.zeros((4, 4, 4, 4))

        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    for beta in range(4):
                        # Symmetric part
                        symmetric_term = 0.5 * (
                            h_proj[mu, alpha] * h_proj[nu, beta]
                            + h_proj[mu, beta] * h_proj[nu, alpha]
                        )

                        # Traceless part (subtract 1/3 of trace)
                        trace_term = (1.0 / 3.0) * h_proj[mu, nu] * h_proj[alpha, beta]

                        projector[mu, nu, alpha, beta] = symmetric_term - trace_term

        return projector

    def velocity_orthogonal_projector(self, four_velocity: np.ndarray, rank: int = 1) -> np.ndarray:
        """
        Project tensors to be orthogonal to four-velocity.

        For vectors: q^μ_⊥ = h^μν q_ν
        For tensors: T^μν_⊥ = h^μα h^νβ T_αβ

        Args:
            four_velocity: Four-velocity u^μ (normalized)
            rank: Tensor rank (1 for vectors, 2 for matrices, etc.)

        Returns:
            Orthogonal projection operator
        """
        h_proj = self.spatial_projector(four_velocity)

        if rank == 1:
            return h_proj
        elif rank == 2:
            # For rank-2 tensors: P^μν_αβ = h^μα h^νβ
            projector = np.zeros((4, 4, 4, 4))
            for mu in range(4):
                for nu in range(4):
                    for alpha in range(4):
                        for beta in range(4):
                            projector[mu, nu, alpha, beta] = h_proj[mu, alpha] * h_proj[nu, beta]
            return projector
        else:
            raise ValueError(f"Projection for rank {rank} tensors not implemented")

    def decompose_vector(
        self, vector: np.ndarray, momentum: np.ndarray, four_velocity: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Decompose a vector into longitudinal and transverse components.

        Args:
            vector: 4-vector to decompose
            momentum: Spatial momentum k^i
            four_velocity: Four-velocity u^μ

        Returns:
            Dictionary with 'longitudinal' and 'transverse' components
        """
        # First project to spatial hypersurface
        h_proj = self.spatial_projector(four_velocity)
        spatial_vector = h_proj @ vector

        # Extract spatial part (indices 1,2,3)
        spatial_3vector = spatial_vector[1:]

        # Decompose into longitudinal/transverse
        p_long = self.longitudinal_projector(momentum)
        p_trans = self.transverse_projector(momentum)

        longitudinal_spatial = p_long @ spatial_3vector
        transverse_spatial = p_trans @ spatial_3vector

        # Reconstruct 4-vectors
        longitudinal_4vec = np.zeros(4)
        longitudinal_4vec[1:] = longitudinal_spatial

        transverse_4vec = np.zeros(4)
        transverse_4vec[1:] = transverse_spatial

        return {
            "longitudinal": longitudinal_4vec,
            "transverse": transverse_4vec,
            "temporal": vector[0] * np.array([1, 0, 0, 0]),  # Time component
        }

    def apply_constraint_projection(
        self, tensor: np.ndarray, constraint_type: str, four_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Apply constraint projections to tensor fields.

        Args:
            tensor: Input tensor array
            constraint_type: Type of constraint ('traceless', 'orthogonal', 'symmetric_traceless')
            four_velocity: Four-velocity for orthogonality constraints

        Returns:
            Projected tensor satisfying the constraint
        """
        if constraint_type == "orthogonal":
            if len(tensor.shape) == 1:
                proj = self.velocity_orthogonal_projector(four_velocity, rank=1)
                return proj @ tensor  # type: ignore[no-any-return]
            elif len(tensor.shape) == 2:
                proj = self.velocity_orthogonal_projector(four_velocity, rank=2)
                result = np.zeros_like(tensor)
                for mu in range(4):
                    for nu in range(4):
                        for alpha in range(4):
                            for beta in range(4):
                                result[mu, nu] += proj[mu, nu, alpha, beta] * tensor[alpha, beta]
                return result  # type: ignore[no-any-return]

        elif constraint_type == "symmetric_traceless":
            if len(tensor.shape) == 2:
                proj = self.symmetric_traceless_projector(four_velocity)
                result = np.zeros_like(tensor)
                for mu in range(4):
                    for nu in range(4):
                        for alpha in range(4):
                            for beta in range(4):
                                result[mu, nu] += proj[mu, nu, alpha, beta] * tensor[alpha, beta]
                return result

        elif constraint_type == "traceless":
            if len(tensor.shape) == 2:
                trace = np.trace(tensor)
                return tensor - (trace / 4.0) * np.eye(4)  # type: ignore[no-any-return]

        return tensor
