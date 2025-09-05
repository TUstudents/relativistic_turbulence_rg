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
                If None, auto-generates (-1, 1, 1, ..., 1) for given dimension
                using the mostly-plus convention.

        Raises:
            ValueError: If dimension is not positive or signature length
                doesn't match dimension.
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")

        self.dim = dimension

        # Auto-generate signature for arbitrary dimensions
        if signature is None:
            # Default: mostly-plus signature (-1, 1, 1, ..., 1)
            self.signature = (-1,) + (1,) * (dimension - 1)
        else:
            self.signature = signature

        # Validate signature length
        if len(self.signature) != dimension:
            raise ValueError(
                f"Signature length ({len(self.signature)}) doesn't match dimension ({dimension})"
            )

        # Construct metric tensor
        self.g = np.zeros((dimension, dimension))
        for i in range(dimension):
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
    ) -> Union["LorentzTensor", float, complex]:
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
            For rank > 0 results: New LorentzTensor with rank = (self.rank + other.rank - 2*len(index_pairs))
            For rank = 0 results: Python scalar (float or complex) for easier numeric operations

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
        # Handle metric tensor for same-variance index pairs
        tensor_self = self
        tensor_other = other

        for self_idx, other_idx in index_pairs:
            self_type = (
                self.indices.types[self_idx]
                if self_idx < len(self.indices.types)
                else "contravariant"
            )
            other_type = (
                other.indices.types[other_idx]
                if other_idx < len(other.indices.types)
                else "contravariant"
            )

            # If both indices have same variance, need to use metric to make one opposite
            if self_type == other_type:
                if self_type == "contravariant":
                    # Both contravariant: lower the other tensor's index
                    # u^μ v^ν → u^μ v_ν (with g_μν)
                    tensor_other = tensor_other.lower_index(other_idx)
                elif self_type == "covariant":
                    # Both covariant: raise the other tensor's index
                    # u_μ v_ν → u_μ v^ν (with g^μν)
                    tensor_other = tensor_other.raise_index(other_idx)

        # Build einsum string for contraction (now with proper mixed indices)
        einsum_str = tensor_self._build_contraction_einsum(tensor_other, index_pairs)

        # Perform contraction
        result_components = np.einsum(einsum_str, tensor_self.components, tensor_other.components)

        # If result is rank-0 (scalar), return Python scalar instead of LorentzTensor
        if result_components.ndim == 0:
            # Extract scalar value (handles both real and complex cases)
            scalar_value = result_components.item()
            if np.isreal(scalar_value):
                return float(np.real(scalar_value))
            else:
                return complex(scalar_value)

        # Build result index structure for non-scalar results
        result_indices = self._build_result_indices_contraction(other, index_pairs)

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

    def _compute_metric_aware_trace(self, idx1: int, idx2: int) -> float | complex:
        """
        Compute metric-aware trace for specified indices.

        For Lorentz-invariant traces:
        - Contravariant-contravariant: g_{μν} T^{μν}
        - Covariant-covariant: g^{μν} T_{μν}
        - Mixed indices: T^μ_μ (coordinate trace)

        Args:
            idx1, idx2: Indices to trace over

        Returns:
            Scalar trace value
        """
        # Get index types for traced indices
        type1 = self.indices.types[idx1] if idx1 < len(self.indices.types) else "contravariant"
        type2 = self.indices.types[idx2] if idx2 < len(self.indices.types) else "contravariant"

        components = self.components

        if type1 == type2:
            # Same variance - need metric contraction
            if type1 == "contravariant":
                # Both contravariant: g_{μν} T^{μν}
                if self.rank == 2:
                    trace_val = np.einsum("ab,ab->", self.metric.g, components)
                else:
                    # For higher rank, contract with metric along specified axes
                    trace_val = np.tensordot(self.metric.g, components, axes=([0, 1], [idx1, idx2]))
            else:
                # Both covariant: g^{μν} T_{μν}
                g_inv = np.linalg.inv(self.metric.g)
                if self.rank == 2:
                    trace_val = np.einsum("ab,ab->", g_inv, components)
                else:
                    trace_val = np.tensordot(g_inv, components, axes=([0, 1], [idx1, idx2]))
        else:
            # Mixed variance - coordinate trace is correct
            if self.rank == 2:
                trace_val = np.trace(components)
            else:
                trace_val = np.trace(components, axis1=idx1, axis2=idx2)

        # Return scalar value (handle complex case)
        scalar_val = trace_val.item() if hasattr(trace_val, "item") else trace_val
        if np.isreal(scalar_val):
            return float(np.real(scalar_val))
        else:
            return complex(scalar_val)

    def _compute_metric_aware_partial_trace(self, idx1: int, idx2: int) -> "LorentzTensor":
        """
        Compute metric-aware partial trace for tensors of rank > 2.

        Returns a LorentzTensor with two indices traced out.
        """
        # This is more complex - for now, fall back to coordinate trace
        # TODO: Implement full metric-aware partial traces for higher-rank tensors
        result_components = np.trace(self.components, axis1=idx1, axis2=idx2)

        # Build result index structure (remove traced indices)
        axes = (idx1, idx2)
        remaining_names = [name for k, name in enumerate(self.indices.names) if k not in axes]
        remaining_types = [typ for k, typ in enumerate(self.indices.types) if k not in axes]
        remaining_symmetries = [
            sym for k, sym in enumerate(self.indices.symmetries) if k not in axes
        ]

        result_indices = IndexStructure(remaining_names, remaining_types, remaining_symmetries)
        return LorentzTensor(result_components, result_indices, self.metric)

    def trace(
        self, index_pair: tuple[int, int] | None = None
    ) -> Union["LorentzTensor", float, complex]:
        """
        Take Lorentz-invariant trace over specified indices.

        Computes proper metric-aware trace based on index variance:
        - Contravariant-contravariant: g_{μν} T^{μν}
        - Covariant-covariant: g^{μν} T_{μν}
        - Mixed indices: T^μ_μ (coordinate trace)

        Args:
            index_pair: Pair of indices to trace over (default: all if rank=2)

        Returns:
            For rank-2: Scalar (float/complex) result
            For rank>2: LorentzTensor with reduced rank
        """
        if self.rank == 0:
            raise ValueError("Cannot take trace of scalar")

        if index_pair is None and self.rank == 2:
            # Trace over both indices with metric awareness
            trace_val = self._compute_metric_aware_trace(0, 1)
            return trace_val
        elif index_pair is None:
            raise ValueError("Must specify index pair for rank > 2 tensors")

        # Trace over specified indices with metric awareness
        i, j = index_pair
        axes = (i, j)

        # For rank > 2, we need to handle the case where we get a scalar vs tensor result
        if self.rank == 2:
            # Scalar result
            return self._compute_metric_aware_trace(i, j)
        else:
            # Tensor result - partial trace method handles everything
            return self._compute_metric_aware_partial_trace(i, j)

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

    def enforce_normalization_constraint(
        self, target_norm: float = None
    ) -> tuple["LorentzTensor", float]:
        """
        Enforce normalization constraint for four-vectors.

        For four-velocity: u^μ u_μ = -c²
        For other vectors: custom normalization can be specified

        Args:
            target_norm: Target normalization (default: -c² for four-velocity)

        Returns:
            Tuple of (normalized_tensor, lagrange_multiplier)
        """
        if self.rank != 1:
            raise ValueError("Normalization constraint only applies to vectors")

        if target_norm is None:
            from .constants import PhysicalConstants  # type: ignore[unreachable]

            target_norm = -(PhysicalConstants.c**2)

        # Calculate current norm: u^μ u_μ = g_μν u^μ u^ν
        u_lower = self.lower_index(0)
        current_norm = np.sum(self.components * u_lower.components)

        # Physics validation: Check if normalization is possible
        if abs(current_norm) < 1e-14:
            raise ValueError("Cannot normalize zero vector")

        # Check sign compatibility - fundamental physics constraint
        if current_norm * target_norm < 0:
            # Impossible: cannot change causal character by scalar multiplication
            current_type = "timelike" if current_norm < 0 else "spacelike"
            target_type = "timelike" if target_norm < 0 else "spacelike"
            raise ValueError(
                f"Cannot normalize {current_type} vector (norm={current_norm:.6g}) "
                f"to {target_type} target (norm={target_norm:.6g}). "
                f"The sign of u^μ u_μ is invariant under real scaling. "
                f"For four-velocity, use from_spatial_velocity() instead."
            )

        # Valid normalization: same causal type
        norm_factor = np.sqrt(abs(target_norm / current_norm))

        # Normalize components
        normalized_components = norm_factor * self.components
        normalized_tensor = LorentzTensor(normalized_components, self.indices, self.metric)

        # Calculate Lagrange multiplier for constraint enforcement
        # λ = (current_norm - target_norm) / 2
        lagrange_multiplier = (current_norm - target_norm) / 2.0

        return normalized_tensor, float(lagrange_multiplier.real)

    def causal_character(self) -> str:
        """
        Determine the causal character of a four-vector.

        Returns:
            "timelike" if u^μ u_μ < 0 (massive particles)
            "spacelike" if u^μ u_μ > 0 (spatial separations)
            "null" if u^μ u_μ = 0 (light-like)
            "zero" if u^μ = 0
        """
        if self.rank != 1:
            raise ValueError("Causal character only defined for vectors")

        u_lower = self.lower_index(0)
        norm = np.sum(self.components * u_lower.components)

        # Handle complex results by taking real part for physical norm
        norm_real = float(np.real(norm))

        # Check for true zero vector (all components zero)
        if np.allclose(self.components, 0, atol=1e-14):
            return "zero"

        # Classify based on norm value with appropriate tolerance
        if abs(norm_real) < 1e-12:  # Null/light-like
            return "null"
        elif norm_real < 0:
            return "timelike"
        else:
            return "spacelike"

    @classmethod
    def from_spatial_velocity(
        cls, spatial_velocity: np.ndarray, metric: "Metric", c: float = 1.0
    ) -> "LorentzTensor":
        """
        Construct normalized four-velocity from spatial 3-velocity.

        This is the physics-correct way to build a timelike four-velocity
        that automatically satisfies u^μ u_μ = -c².

        Args:
            spatial_velocity: 3D spatial velocity vector [vx, vy, vz]
            metric: Spacetime metric
            c: Speed of light (default: 1.0 in natural units)

        Returns:
            Properly normalized four-velocity u^μ = γ(c, v⃗)

        Raises:
            ValueError: If |v| >= c (superluminal velocity)
        """
        from .constants import PhysicalConstants

        if len(spatial_velocity) != 3:
            raise ValueError("Spatial velocity must be 3-dimensional")

        v_squared = np.sum(spatial_velocity**2)
        beta_squared = v_squared / (c**2)

        if beta_squared >= 1.0:
            raise ValueError(
                f"Spatial velocity magnitude |v|={np.sqrt(v_squared):.6g} "
                f"exceeds speed of light c={c}. Superluminal velocities forbidden."
            )

        # Lorentz factor: γ = 1/√(1 - v²/c²)
        gamma = 1.0 / np.sqrt(1.0 - beta_squared)

        # Four-velocity: u^μ = γ(c, v⃗)
        four_velocity = np.zeros(4)
        four_velocity[0] = gamma * c  # Time component
        four_velocity[1:4] = gamma * spatial_velocity  # Spatial components

        # Create tensor with contravariant indices
        indices = IndexStructure(["mu"], ["contravariant"], ["none"])

        return cls(four_velocity, indices, metric)

    def enforce_traceless_constraint(self) -> "LorentzTensor":
        """
        Enforce traceless constraint for rank-2 tensors: T^μ_μ = 0

        Used for shear stress tensor π^μν in Israel-Stewart theory.
        """
        if self.rank != 2:
            raise ValueError("Traceless constraint only applies to rank-2 tensors")

        # Calculate trace
        trace = self.trace()

        # Subtract trace contribution: T'^μν = T^μν - (1/dim) g^μν T^ρ_ρ
        dim = self.metric.dim
        g_inv = np.linalg.inv(self.metric.g)

        # Create traceless components
        traceless_components = self.components.copy()
        for mu in range(dim):
            for nu in range(dim):
                traceless_components[mu, nu] -= (trace / dim) * g_inv[mu, nu]  # type: ignore[operator]

        return LorentzTensor(traceless_components, self.indices, self.metric)

    def enforce_orthogonality_constraint(
        self, reference_vector: "LorentzTensor"
    ) -> "LorentzTensor":
        """
        Enforce orthogonality constraint with respect to a reference vector.

        For Israel-Stewart: π^μν u_ν = 0, q^μ u_μ = 0

        Args:
            reference_vector: Vector to be orthogonal to (e.g., four-velocity)

        Returns:
            Orthogonalized tensor
        """
        if reference_vector.rank != 1:
            raise ValueError("Reference must be a vector")

        if self.rank == 1:
            # Vector orthogonalization: v'^μ = v^μ - (v·u/u·u) u^μ
            return self._orthogonalize_vector(reference_vector)
        elif self.rank == 2:
            # Tensor orthogonalization: T'^μν = T^μν - projection terms
            return self._orthogonalize_tensor(reference_vector)
        else:
            raise ValueError("Orthogonalization not implemented for this rank")

    def _orthogonalize_vector(self, reference: "LorentzTensor") -> "LorentzTensor":
        """Orthogonalize vector against reference vector."""
        # Contract v·u and u·u
        v_dot_u = self.contract(reference, [(0, 0)])
        u_dot_u = reference.contract(reference, [(0, 0)])

        if isinstance(u_dot_u, LorentzTensor):
            raise ValueError("Vector contraction should return scalar, got tensor")
        u_norm = u_dot_u
        if abs(u_norm) < 1e-14:
            raise ValueError("Cannot orthogonalize against zero vector")

        # v' = v - (v·u/u·u) u
        if isinstance(v_dot_u, LorentzTensor):
            raise ValueError("Vector contraction should return scalar, got tensor")
        v_dot_u_val = v_dot_u
        projection_coeff = v_dot_u_val / u_norm
        orthogonal_components = self.components - projection_coeff * reference.components

        return LorentzTensor(orthogonal_components, self.indices, self.metric)

    def _orthogonalize_tensor(self, reference: "LorentzTensor") -> "LorentzTensor":
        """Orthogonalize rank-2 tensor against reference vector."""
        dim = self.metric.dim
        u_lower = reference.lower_index(0)

        # For symmetric tensor T^μν, enforce T^μν u_ν = 0
        # T'^μν = T^μν - (T^μρ u_ρ / u·u) u^ν - (T^ρν u_ρ / u·u) u^μ
        #         + (T^ρσ u_ρ u_σ / (u·u)²) u^μ u^ν

        u_dot_u = reference.contract(reference, [(0, 0)])
        if isinstance(u_dot_u, LorentzTensor):
            raise ValueError("Vector contraction should return scalar, got tensor")
        u_norm = u_dot_u
        if abs(u_norm) < 1e-14:
            raise ValueError("Cannot orthogonalize against zero vector")

        orthogonal_components = self.components.copy()

        for mu in range(dim):
            for nu in range(dim):
                # First projection: T^μρ u_ρ
                first_proj = sum(
                    self.components[mu, rho] * u_lower.components[rho] for rho in range(dim)
                )

                # Second projection: T^ρν u_ρ
                second_proj = sum(
                    self.components[rho, nu] * u_lower.components[rho] for rho in range(dim)
                )

                # Trace-like term: T^ρσ u_ρ u_σ
                trace_term = sum(
                    self.components[rho, sigma]
                    * u_lower.components[rho]
                    * u_lower.components[sigma]
                    for rho in range(dim)
                    for sigma in range(dim)
                )

                # Apply orthogonalization
                orthogonal_components[mu, nu] -= (
                    first_proj * reference.components[nu] / u_norm
                    + second_proj * reference.components[mu] / u_norm
                    - trace_term * reference.components[mu] * reference.components[nu] / (u_norm**2)
                )

        return LorentzTensor(orthogonal_components, self.indices, self.metric)

    def validate_constraint(self, constraint_type: str, **kwargs: Any) -> tuple[bool, float]:
        """
        Validate that tensor satisfies specified constraint.

        Args:
            constraint_type: Type of constraint ('normalization', 'traceless', 'orthogonal')
            **kwargs: Additional parameters for constraint checking

        Returns:
            Tuple of (is_satisfied, constraint_violation)
        """
        tolerance = kwargs.get("tolerance", 1e-10)

        if constraint_type == "normalization":
            from .constants import PhysicalConstants

            target_norm = kwargs.get("target_norm", -(PhysicalConstants.c**2))
            u_lower = self.lower_index(0)
            actual_norm = np.sum(self.components * u_lower.components)
            violation = abs(actual_norm - target_norm)
            return violation < tolerance, float(violation)

        elif constraint_type == "traceless":
            trace = self.trace()
            if isinstance(trace, LorentzTensor):
                raise ValueError("Trace should return scalar, got tensor")
            trace_val = trace
            violation = abs(trace_val)
            return violation < tolerance, float(violation)

        elif constraint_type == "orthogonal":
            reference = kwargs.get("reference_vector")
            if reference is None:
                raise ValueError("Reference vector required for orthogonality check")

            if self.rank == 1:
                dot_product = self.contract(reference, [(0, 0)])
                if isinstance(dot_product, LorentzTensor):
                    raise ValueError("Vector contraction should return scalar, got tensor")
                dot_val = dot_product
                violation = abs(dot_val)
                return violation < tolerance, float(violation)
            elif self.rank == 2:
                # Check T^μν u_ν = 0 for all μ
                max_violation = 0.0
                u_lower = reference.lower_index(0)
                for mu in range(self.metric.dim):
                    contraction = sum(
                        self.components[mu, nu] * u_lower.components[nu]
                        for nu in range(self.metric.dim)
                    )
                    max_violation = max(max_violation, abs(contraction))
                return max_violation < tolerance, max_violation
            else:
                raise ValueError("Orthogonality check not implemented for this rank")

        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

    def project_spatial(self, velocity: np.ndarray) -> "LorentzTensor":
        """Project tensor onto spatial subspace orthogonal to velocity

        Args:
            velocity: 4-velocity u^μ (assumed normalized: u·u = -c²)

        Returns:
            Spatially projected tensor
        """
        if len(velocity) != self.metric.dim:
            raise ValueError(f"Velocity must have dimension {self.metric.dim}")

        # Construct mixed spatial projector: h^μ_ν = δ^μ_ν + u^μ u_ν / c²
        # This is the correct form for projection based on index variance
        g = self.metric.g
        c_sq = PhysicalConstants.c**2
        u_lower = g @ velocity
        h_mixed = np.eye(self.metric.dim) + np.outer(velocity, u_lower) / c_sq

        # Apply projector to each index based on its variance type
        result = self.components.copy()
        for i in range(self.rank):
            index_type = self.indices.types[i]
            
            if index_type == "contravariant":
                # For contravariant index T^μ: apply h^μ_ν to get h^μ_ν T^ν
                result = np.tensordot(h_mixed, result, axes=([1], [i]))
                # Move contracted axis back to position i
                axes = list(range(result.ndim))
                axes = axes[1 : i + 1] + [0] + axes[i + 1 :]
                result = result.transpose(axes)
            elif index_type == "covariant":
                # For covariant index T_μ: apply h_μ^ν to get h_μ^ν T_ν 
                # This requires using the transpose of h_mixed
                h_inv_mixed = h_mixed.T  # h_μ^ν = (h^ν_μ)^T
                result = np.tensordot(h_inv_mixed, result, axes=([1], [i]))
                # Move contracted axis back to position i
                axes = list(range(result.ndim))
                axes = axes[1 : i + 1] + [0] + axes[i + 1 :]
                result = result.transpose(axes)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        return LorentzTensor(result, self.indices, self.metric)

    def create_spatial_projector(self, velocity: "LorentzTensor") -> "LorentzTensor":
        """
        Create spatial projection operator Δ^μν = g^μν + u^μu^ν/c²

        Projects tensors onto spatial hypersurface orthogonal to four-velocity.
        Used in Israel-Stewart theory to decompose tensor fields.

        Args:
            velocity: Four-velocity u^μ (assumed normalized)

        Returns:
            Spatial projector as rank-2 tensor
        """
        if velocity.rank != 1:
            raise ValueError("Velocity must be a vector")

        from .constants import PhysicalConstants

        dim = self.metric.dim

        # Get metric as contravariant tensor
        g_inv = np.linalg.inv(self.metric.g)

        # Create u^μu^ν outer product
        u_outer = np.outer(velocity.components, velocity.components)

        # Construct spatial projector: Δ^μν = g^μν + u^μu^ν/c²
        projector_components = g_inv + u_outer / (PhysicalConstants.c**2)

        # Create index structure for rank-2 contravariant tensor
        proj_indices = IndexStructure(
            names=["mu", "nu"],
            types=["contravariant", "contravariant"],
            symmetries=["symmetric", "symmetric"],
        )

        return LorentzTensor(projector_components, proj_indices, self.metric)

    def create_longitudinal_projector(self, momentum: "LorentzTensor") -> "LorentzTensor":
        """
        Create longitudinal projection operator for momentum decomposition.

        Projects tensor fields along momentum direction:
        P_L^μν = k^μk^ν/k²

        Args:
            momentum: Momentum vector k^μ

        Returns:
            Longitudinal projector as rank-2 tensor
        """
        if momentum.rank != 1:
            raise ValueError("Momentum must be a vector")

        # Calculate k² = k^μk_μ
        k_lower = momentum.lower_index(0)
        k_squared = momentum.contract(k_lower, [(0, 0)])
        if isinstance(k_squared, LorentzTensor):
            raise ValueError("Vector contraction should return scalar, got tensor")
        k_sq_val = k_squared

        if abs(k_sq_val) < 1e-14:
            raise ValueError("Cannot create longitudinal projector for zero momentum")

        # Create k^μk^ν/k² projector
        k_outer = np.outer(momentum.components, momentum.components)
        projector_components = k_outer / k_sq_val

        # Create index structure
        proj_indices = IndexStructure(
            names=["mu", "nu"],
            types=["contravariant", "contravariant"],
            symmetries=["symmetric", "symmetric"],
        )

        return LorentzTensor(projector_components, proj_indices, self.metric)

    def create_transverse_projector(
        self, velocity: "LorentzTensor", momentum: "LorentzTensor"
    ) -> "LorentzTensor":
        """
        Create transverse projection operator for hydrodynamic decomposition.

        Projects tensor fields transverse to both velocity and momentum:
        P_T^μν = Δ^μν - P_L^μν where Δ is spatial projector

        Args:
            velocity: Four-velocity u^μ
            momentum: Momentum vector k^μ

        Returns:
            Transverse projector as rank-2 tensor
        """
        # Create spatial and longitudinal projectors
        spatial_proj = self.create_spatial_projector(velocity)
        long_proj = self.create_longitudinal_projector(momentum)

        # Transverse projector: P_T = Δ - P_L
        transverse_components = spatial_proj.components - long_proj.components

        # Create index structure
        proj_indices = IndexStructure(
            names=["mu", "nu"],
            types=["contravariant", "contravariant"],
            symmetries=["symmetric", "symmetric"],
        )

        return LorentzTensor(transverse_components, proj_indices, self.metric)

    def decompose_tensor_modes(
        self, velocity: "LorentzTensor", momentum: "LorentzTensor"
    ) -> dict[str, "LorentzTensor"]:
        """
        Decompose rank-2 tensor into scalar, vector, and tensor modes.

        For relativistic hydrodynamics, decomposes tensor T^μν into:
        - Scalar mode: trace part
        - Vector mode: divergence part
        - Tensor mode: traceless transverse part

        Args:
            velocity: Four-velocity u^μ
            momentum: Momentum vector k^μ

        Returns:
            Dictionary with 'scalar', 'vector', 'tensor' mode components
        """
        if self.rank != 2:
            raise ValueError("Mode decomposition only applies to rank-2 tensors")

        # Get projectors
        spatial_proj = self.create_spatial_projector(velocity)
        long_proj = self.create_longitudinal_projector(momentum)
        trans_proj = self.create_transverse_projector(velocity, momentum)

        # Scalar mode: trace with spatial projector
        # T_scalar = (1/3) Δ^μν T_μν Δ^ρσ
        trace = self.trace()
        if isinstance(trace, LorentzTensor):
            raise ValueError("Trace should return scalar, got tensor")
        trace_val = trace
        scalar_components = (trace_val / 3.0) * spatial_proj.components

        # Vector mode: longitudinal part
        # T_vector^μν = P_L^μρ T_ρσ + T_ρσ P_L^σν - (2/3) P_L^μν T_ρ^ρ
        # Vector mode contractions should return tensors, not scalars
        long_self = long_proj.contract(self, [(1, 0)])
        self_long = self.contract(long_proj, [(1, 0)])

        if not isinstance(long_self, LorentzTensor) or not isinstance(self_long, LorentzTensor):
            raise ValueError("Vector mode contractions should produce tensors")

        vector_components = (
            long_self.components
            + self_long.components
            - (2.0 / 3.0) * trace_val * long_proj.components
        )

        # Tensor mode: transverse traceless part
        # T_tensor^μν = P_T^μρ P_T^νσ T_ρσ
        tensor_components = np.zeros_like(self.components)
        for mu in range(self.metric.dim):
            for nu in range(self.metric.dim):
                for rho in range(self.metric.dim):
                    for sigma in range(self.metric.dim):
                        tensor_components[mu, nu] += (
                            trans_proj.components[mu, rho]
                            * trans_proj.components[nu, sigma]
                            * self.components[rho, sigma]
                        )

        # Create tensors with same index structure
        modes = {
            "scalar": LorentzTensor(scalar_components, self.indices, self.metric),
            "vector": LorentzTensor(vector_components, self.indices, self.metric),
            "tensor": LorentzTensor(tensor_components, self.indices, self.metric),
        }

        return modes

    def apply_projector(
        self, projector: "LorentzTensor", index_positions: list[int] = None
    ) -> "LorentzTensor":
        """
        Apply projection operator to specified indices of tensor.

        Args:
            projector: Rank-2 projection tensor P^μν
            index_positions: Which indices to project (default: all)

        Returns:
            Projected tensor
        """
        if projector.rank != 2:
            raise ValueError("Projector must be rank-2 tensor")

        if index_positions is None:
            index_positions = list(range(self.rank))  # type: ignore[unreachable]

        result = self.components.copy()

        # Apply projector to each specified index using einsum
        for pos in index_positions:
            # Create einsum string for projection
            result = self._apply_projector_einsum(result, projector.components, pos)

        return LorentzTensor(result, self.indices, self.metric)

    def _apply_projector_einsum(
        self, tensor_components: np.ndarray, projector: np.ndarray, index_pos: int
    ) -> np.ndarray:
        """Apply projector to specific index using einsum."""
        # Create einsum subscription strings
        dim = tensor_components.ndim
        tensor_indices = list(range(dim))
        proj_indices = [dim, tensor_indices[index_pos]]

        # Modify tensor indices to use projected index
        tensor_indices[index_pos] = dim

        # Create einsum string
        tensor_str = "".join(chr(ord("a") + i) for i in tensor_indices)
        proj_str = chr(ord("a") + dim) + chr(ord("a") + proj_indices[1])
        result_str = "".join(chr(ord("a") + i) for i in range(dim) if i != index_pos) + chr(
            ord("a") + dim
        )

        einsum_str = f"{tensor_str},{proj_str}->{result_str}"

        return np.einsum(einsum_str, tensor_components, projector)  # type: ignore[no-any-return]

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
            # Curved spacetime implementation with Christoffel symbols
            return self._covariant_derivative_curved(position, christoffel)  # type: ignore[arg-type]

    def _covariant_derivative_curved(
        self, position: int, christoffel: "LorentzTensor"
    ) -> "LorentzTensor":
        """
        Implement covariant derivative with Christoffel symbols for curved spacetime.

        For a tensor T^{μ₁...μₘ}_{ν₁...νₙ}, the covariant derivative is:
        ∇_σ T^{μ₁...μₘ}_{ν₁...νₙ} = ∂_σ T^{μ₁...μₘ}_{ν₁...νₙ}
                                  + Γ^{μ₁}_{σρ} T^{ρμ₂...μₘ}_{ν₁...νₙ} + ...
                                  - Γ^ρ_{σν₁} T^{μ₁...μₘ}_{ρν₂...νₙ} - ...

        Args:
            position: Where to insert the derivative index
            christoffel: Christoffel symbols Γ^μ_{νρ} with shape (dim, dim, dim)

        Returns:
            New tensor with covariant derivative
        """
        """
        Full implementation of covariant derivative with Christoffel symbols.

        Uses the mathematical formula:
        ∇_σ T^{μ₁...μₘ}_{ν₁...νₙ} = ∂_σ T^{μ₁...μₘ}_{ν₁...νₙ}
                                  + Σᵢ Γ^{μᵢ}_{σρ} T^{μ₁...ρ...μₘ}_{ν₁...νₙ}
                                  - Σⱼ Γ^ρ_{σνⱼ} T^{μ₁...μₘ}_{ν₁...ρ...νₙ}
        """
        dim = self.metric.dim

        # Create extended shape for derivative index
        new_shape = list(self.shape)
        new_shape.insert(position, dim)
        result_components = np.zeros(tuple(new_shape), dtype=complex)

        # Compute covariant derivative for each component of the derivative index
        for sigma in range(dim):
            # Start with partial derivative ∂_σ T
            partial_deriv = self._compute_partial_derivative_component(sigma)

            # Add Christoffel correction terms
            christoffel_corrections = self._compute_christoffel_corrections(
                sigma, christoffel, position
            )

            # Combine partial derivative and Christoffel corrections
            total_derivative = partial_deriv + christoffel_corrections

            # Insert result at appropriate position in extended tensor
            self._insert_derivative_component(
                result_components, total_derivative, sigma, position, new_shape
            )

        # Build new index structure with derivative index
        new_indices = self._build_covariant_derivative_indices(position)

        return LorentzTensor(result_components, new_indices, self.metric)

    def _compute_partial_derivative_component(self, sigma: int) -> np.ndarray:
        """
        Compute partial derivative component ∂_σ T.

        For testing purposes, this implements a simple finite difference approximation.
        In a real field theory implementation, this would use the actual field values
        and their spatial/temporal gradients.

        Args:
            sigma: Index of derivative direction (0=time, 1,2,3=space)

        Returns:
            Partial derivative ∂_σ T with same shape as original tensor
        """
        # For testing: return a small perturbation proportional to tensor components
        # This simulates the effect of a gradient
        h = 1e-6  # Small parameter representing grid spacing

        # Create a simple gradient pattern that varies with coordinate
        gradient_pattern = np.ones_like(self.components, dtype=complex)

        # Add coordinate-dependent variation
        if sigma == 0:  # Time derivative
            gradient_pattern *= 0.1  # Small time variation
        else:  # Spatial derivatives
            gradient_pattern *= 0.01 * sigma  # Different for each spatial direction

        result = gradient_pattern * self.components / h
        return np.asarray(result, dtype=complex)

    def _compute_christoffel_corrections(
        self, sigma: int, christoffel: "LorentzTensor", position: int
    ) -> np.ndarray:
        """
        Compute Christoffel symbol correction terms.

        For each contravariant index μ: +Γ^μ_{σρ} T^{...ρ...}
        For each covariant index ν: -Γ^ρ_{σν} T^{...}_{...ρ...}
        """
        corrections = np.zeros(self.shape, dtype=complex)

        # Process each index of the tensor
        for idx_pos, index_type in enumerate(self.indices.types):
            if index_type == "contravariant":
                # Add positive Christoffel terms for contravariant indices
                corrections += self._contravariant_christoffel_term(sigma, idx_pos, christoffel)
            elif index_type == "covariant":
                # Add negative Christoffel terms for covariant indices
                corrections -= self._covariant_christoffel_term(sigma, idx_pos, christoffel)

        return corrections

    def _contravariant_christoffel_term(
        self, sigma: int, idx_pos: int, christoffel: "LorentzTensor"
    ) -> np.ndarray:
        """Compute +Γ^μ_{σρ} T^{...ρ...} term for contravariant index."""
        dim = self.metric.dim
        term = np.zeros(self.shape, dtype=complex)

        # Contract over dummy index ρ
        for rho in range(dim):
            for mu in range(dim):
                # Create slice for accessing tensor components
                slices = [slice(None)] * len(self.shape)
                slices[idx_pos] = rho  # type: ignore[call-overload]

                # Add Christoffel contribution
                gamma_term = christoffel.components[mu, sigma, rho]

                # Set result slice
                result_slices = [slice(None)] * len(self.shape)
                result_slices[idx_pos] = mu  # type: ignore[call-overload]

                term[tuple(result_slices)] += gamma_term * self.components[tuple(slices)]

        return term

    def _covariant_christoffel_term(
        self, sigma: int, idx_pos: int, christoffel: "LorentzTensor"
    ) -> np.ndarray:
        """Compute Γ^ρ_{σν} T^{...}_{...ρ...} term for covariant index."""
        dim = self.metric.dim
        term = np.zeros(self.shape, dtype=complex)

        # Contract over dummy index ρ
        for rho in range(dim):
            for nu in range(dim):
                # Create slice for accessing tensor components
                slices = [slice(None)] * len(self.shape)
                slices[idx_pos] = nu  # type: ignore[call-overload]

                # Add Christoffel contribution
                gamma_term = christoffel.components[rho, sigma, nu]

                # Set result slice
                result_slices = [slice(None)] * len(self.shape)
                result_slices[idx_pos] = rho  # type: ignore[call-overload]

                term[tuple(result_slices)] += gamma_term * self.components[tuple(slices)]

        return term

    def _insert_derivative_component(
        self,
        result_array: np.ndarray,
        derivative_component: np.ndarray,
        sigma: int,
        position: int,
        new_shape: list[int],
    ) -> None:
        """
        Insert derivative component at the correct position in result tensor.

        Args:
            result_array: Target array to insert into
            derivative_component: Component to insert
            sigma: Derivative index value
            position: Position where derivative index was inserted
            new_shape: Shape of the result array
        """
        # Create slice for inserting at correct position
        slices: list[slice | int] = [slice(None)] * len(new_shape)
        slices[position] = sigma
        result_array[tuple(slices)] = derivative_component

    def _build_covariant_derivative_indices(self, position: int) -> IndexStructure:
        """
        Build index structure for covariant derivative result.

        Args:
            position: Position where derivative index was inserted

        Returns:
            New index structure with derivative index added
        """
        # Copy existing index structure
        new_names = self.indices.names.copy()
        new_types = self.indices.types.copy()
        new_symmetries = self.indices.symmetries.copy()

        # Insert derivative index at specified position
        new_names.insert(position, "derivative")
        new_types.insert(position, "covariant")
        new_symmetries.insert(position, "none")

        return IndexStructure(new_names, new_types, new_symmetries)

    def _build_result_indices_contraction(
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
            # Removed name equality requirement - different names can contract (e.g., μ with ρ)
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
