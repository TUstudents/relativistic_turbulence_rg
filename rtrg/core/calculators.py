"""
Unified calculator framework for physics calculations.

This module provides the base classes and interfaces for all physics
calculators in the relativistic turbulence RG analysis, establishing
consistent patterns for propagator calculations, validation, and analysis.

Key Classes:
    AbstractCalculator: Base class for all physics calculators
    PropagatorCalculatorBase: Base for propagator calculations
    ValidatorBase: Base for physics validation
    CalculationResult: Standardized result container

Features:
    - Consistent initialization and interface patterns
    - Unified error handling and logging
    - Standard result caching mechanisms
    - Common validation framework
    - Type safety with abstract base classes

Usage:
    >>> from rtrg.core.calculators import PropagatorCalculatorBase
    >>> class MyCalculator(PropagatorCalculatorBase):
    ...     def calculate_retarded_propagator(self, field1, field2): ...
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar, cast

import sympy as sp
from sympy import Symbol, symbols

from .fields import Field


class CalculationType(Enum):
    """Enumeration of supported calculation types."""

    PROPAGATOR = "propagator"
    VERTEX = "vertex"
    BETA_FUNCTION = "beta_function"
    VALIDATION = "validation"
    ANALYSIS = "analysis"


@dataclass
class CalculationResult:
    """
    Standardized result container for physics calculations.

    Provides consistent result format across all calculator types
    with metadata, caching support, and error handling.
    """

    result: Any
    calculation_type: CalculationType
    metadata: dict[str, Any] = field(default_factory=dict)
    is_cached: bool = False
    computation_time: float = 0.0
    validation_status: bool = True
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.error_message is not None:
            self.validation_status = False


T = TypeVar("T")


class AbstractCalculator(ABC, Generic[T]):
    """
    Abstract base class for all physics calculators.

    Establishes consistent interface patterns, error handling,
    and caching mechanisms across all calculator implementations.

    Features:
        - Standard initialization with metric and parameters
        - Unified result caching with configurable keys
        - Common symbolic variable management
        - Consistent error handling and logging
        - Type-safe result containers
    """

    def __init__(
        self, metric: Any | None = None, temperature: float = 1.0, enable_caching: bool = True
    ):
        """
        Initialize base calculator.

        Args:
            metric: Spacetime metric tensor (optional)
            temperature: System temperature in natural units
            enable_caching: Whether to cache calculation results
        """
        self.metric = metric
        self.temperature = temperature
        self.enable_caching = enable_caching

        # Unified result cache
        self.result_cache: dict[str, CalculationResult] = {}

        # Common symbolic variables
        self.omega = symbols("omega", complex=True)
        self.k = symbols("k", real=True, positive=True)
        self.k_vec = [symbols(f"k_{i}", real=True) for i in range(3)]

        # Performance tracking
        self._computation_stats: dict[str, int] = {}

    @abstractmethod
    def get_calculation_type(self) -> CalculationType:
        """Return the type of calculations this calculator performs."""
        pass

    def _create_cache_key(self, method_name: str, *args: Any, **kwargs: Any) -> str:
        """
        Create standardized cache key for results.

        Args:
            method_name: Name of the calculation method
            *args: Positional arguments to the method
            **kwargs: Keyword arguments to the method

        Returns:
            Standardized cache key string
        """
        # Convert complex objects to hashable representations
        hashable_args = []
        for arg in args:
            if hasattr(arg, "name"):  # Field objects
                hashable_args.append(arg.name)
            elif isinstance(arg, (int, float, complex, str)):
                hashable_args.append(str(arg))
            else:
                hashable_args.append(str(type(arg).__name__))

        hashable_kwargs = []
        for k, v in kwargs.items():
            if isinstance(v, (int, float, complex, str)):
                hashable_kwargs.append(f"{k}={v}")
            else:
                hashable_kwargs.append(f"{k}={type(v).__name__}")

        args_str = "_".join(hashable_args)
        kwargs_str = "_".join(hashable_kwargs)

        return f"{method_name}_{args_str}_{kwargs_str}"

    def _get_cached_result(self, cache_key: str) -> CalculationResult | None:
        """Get cached result if available and caching is enabled."""
        if not self.enable_caching:
            return None
        return self.result_cache.get(cache_key)

    def _cache_result(self, cache_key: str, result: CalculationResult) -> None:
        """Cache calculation result if caching is enabled."""
        if self.enable_caching:
            result.is_cached = False  # Mark as fresh calculation
            self.result_cache[cache_key] = result

    def _track_computation(self, method_name: str) -> None:
        """Track computation statistics."""
        self._computation_stats[method_name] = self._computation_stats.get(method_name, 0) + 1

    def get_computation_stats(self) -> dict[str, int]:
        """Get computation statistics."""
        return self._computation_stats.copy()

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.result_cache.clear()

    def validate_inputs(self, *args: Any, **kwargs: Any) -> bool:
        """
        Validate inputs for calculations.

        Default implementation performs basic type checking.
        Subclasses should override for domain-specific validation.
        """
        # Basic validation - check for None values in critical parameters
        for arg in args:
            if arg is None:
                warnings.warn("None value passed to calculation method")
                return False
        return True

    @abstractmethod
    def _perform_calculation(self, *args: Any, **kwargs: Any) -> T:
        """
        Perform the actual calculation.

        This method contains the core calculation logic and must be
        implemented by all concrete calculator classes.
        """
        pass

    def calculate(self, *args: Any, **kwargs: Any) -> CalculationResult:
        """
        Main calculation entry point with caching and error handling.

        This method provides a standardized interface for all calculations,
        handling caching, validation, error recovery, and result formatting.
        """
        import time

        # Validate inputs
        if not self.validate_inputs(*args, **kwargs):
            return CalculationResult(
                result=None,
                calculation_type=self.get_calculation_type(),
                error_message="Input validation failed",
            )

        # Check cache
        cache_key = self._create_cache_key("calculate", *args, **kwargs)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            cached_result.is_cached = True
            return cached_result

        # Perform calculation with timing
        start_time = time.time()
        try:
            result = self._perform_calculation(*args, **kwargs)
            computation_time = time.time() - start_time

            # Create result container
            calc_result = CalculationResult(
                result=result,
                calculation_type=self.get_calculation_type(),
                computation_time=computation_time,
                validation_status=True,
            )

            # Cache and track
            self._cache_result(cache_key, calc_result)
            self._track_computation("calculate")

            return calc_result

        except Exception as e:
            computation_time = time.time() - start_time
            warnings.warn(f"Calculation failed: {str(e)}")

            return CalculationResult(
                result=None,
                calculation_type=self.get_calculation_type(),
                computation_time=computation_time,
                error_message=str(e),
            )

    def __str__(self) -> str:
        """String representation with calculation statistics."""
        calc_type = self.get_calculation_type().value
        cached_count = len(self.result_cache)
        total_computations = sum(self._computation_stats.values())
        return f"{self.__class__.__name__}(type={calc_type}, cached={cached_count}, computed={total_computations})"


class ValidatorBase(AbstractCalculator[bool]):
    """
    Base class for physics validation calculators.

    Provides standardized validation interface with consistent
    result reporting and error handling.
    """

    def get_calculation_type(self) -> CalculationType:
        """Validators perform validation calculations."""
        return CalculationType.VALIDATION

    @dataclass
    class ValidationResult:
        """Specialized result container for validation."""

        is_valid: bool
        violations: list[str] = field(default_factory=list)
        metrics: dict[str, float] = field(default_factory=dict)
        tolerance: float = 1e-12

        def add_violation(self, violation: str, metric_value: float = 0.0) -> None:
            """Add a validation violation with optional metric."""
            self.violations.append(violation)
            if violation not in self.metrics:
                self.metrics[violation] = metric_value
            self.is_valid = False

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> "ValidatorBase.ValidationResult":
        """
        Perform validation check.

        Must be implemented by concrete validator classes.
        """
        pass

    def _perform_calculation(self, *args: Any, **kwargs: Any) -> bool:
        """Perform validation and return boolean result."""
        validation_result = self.validate(*args, **kwargs)
        return validation_result.is_valid


# Import here to avoid circular dependencies
from dataclasses import dataclass as dc


@dc
class PropagatorComponents:
    """Container for different components of a propagator."""

    retarded: sp.Expr | None = None
    advanced: sp.Expr | None = None
    keldysh: sp.Expr | None = None
    spectral: sp.Expr | None = None

    def __post_init__(self) -> None:
        """Validate propagator components."""
        if self.retarded is not None and self.advanced is not None:
            # Could add causality relation checks here
            pass


@dataclass
class SpectralProperties:
    """Container for spectral properties of propagators."""

    poles: list[complex] = field(default_factory=list)
    residues: list[complex] = field(default_factory=list)
    spectral_weight: float | None = None
    causality_satisfied: bool = True

    def validate_causality(self) -> bool:
        """Validate causality structure (poles in lower half-plane for retarded)."""
        for pole in self.poles:
            if pole.imag >= 0:  # Pole in upper half-plane violates causality
                self.causality_satisfied = False
                return False
        return True


@dc
class PropagatorMatrix:
    """Container for propagator matrices in field space."""

    matrix: sp.Matrix
    field_names: list[str] = field(default_factory=list)
    matrix_type: str = "retarded"  # "retarded", "advanced", "keldysh"

    def get_component(self, field1: str, field2: str) -> sp.Expr | None:
        """Get propagator component between two fields."""
        try:
            i = self.field_names.index(field1)
            j = self.field_names.index(field2)
            return self.matrix[i, j]
        except (ValueError, IndexError):
            return None

    def invert(self) -> "PropagatorMatrix":
        """Return inverted propagator matrix."""
        try:
            inverted_matrix = self.matrix.inv()
            return PropagatorMatrix(
                matrix=inverted_matrix,
                field_names=self.field_names.copy(),
                matrix_type=f"inverse_{self.matrix_type}",
            )
        except Exception as e:
            warnings.warn(f"Matrix inversion failed: {str(e)}")
            # Return identity matrix as fallback
            identity = sp.eye(len(self.field_names))
            return PropagatorMatrix(
                matrix=identity,
                field_names=self.field_names.copy(),
                matrix_type="identity_fallback",
            )


class PropagatorCalculatorBase(AbstractCalculator[PropagatorComponents]):
    """
    Abstract base class for all propagator calculators.

    Establishes consistent interface for propagator calculations across
    different implementation strategies (simple, enhanced, tensor-aware).

    Standard Interface:
        - calculate_retarded_propagator()
        - calculate_advanced_propagator()
        - calculate_keldysh_propagator()
        - calculate_spectral_function()
        - extract_poles()

    Features:
        - Common symbolic variable management (omega, k, k_vec)
        - Unified caching with field-pair keys
        - Standard result format (PropagatorComponents)
        - Causality validation
        - FDT relation checking
    """

    def __init__(
        self,
        msrjd_action: Any,
        metric: Any | None = None,
        temperature: float = 1.0,
        enable_caching: bool = True,
    ):
        """
        Initialize propagator calculator base.

        Args:
            msrjd_action: MSRJD action instance for propagator extraction
            metric: Spacetime metric tensor
            temperature: System temperature for FDT relations
            enable_caching: Whether to enable result caching
        """
        super().__init__(metric, temperature, enable_caching)

        self.action = msrjd_action
        self.is_system = msrjd_action.is_system

        # Specialized caches for propagator calculations
        self.propagator_cache: dict[str, PropagatorComponents] = {}
        self.matrix_cache: dict[str, PropagatorMatrix] = {}
        self.spectral_cache: dict[str, SpectralProperties] = {}

    def get_calculation_type(self) -> CalculationType:
        """Propagator calculators perform propagator calculations."""
        return CalculationType.PROPAGATOR

    def _create_propagator_cache_key(
        self, method_name: str, field1: Field, field2: Field, **kwargs: Any
    ) -> str:
        """Create cache key for propagator calculations."""
        base_key = f"{method_name}_{field1.name}_{field2.name}"

        # Add parameter-specific suffixes
        if "omega_val" in kwargs and kwargs["omega_val"] is not None:
            base_key += f"_omega_{kwargs['omega_val']}"
        if "k_val" in kwargs and kwargs["k_val"] is not None:
            base_key += f"_k_{kwargs['k_val']}"

        return base_key

    @abstractmethod
    def calculate_retarded_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """
        Calculate retarded propagator G^R_{field1,field2}(ω,k).

        Args:
            field1: First field for propagator calculation
            field2: Second field for propagator calculation
            omega_val: Specific frequency value (optional)
            k_val: Specific momentum value (optional)

        Returns:
            Symbolic expression for retarded propagator
        """
        pass

    @abstractmethod
    def calculate_advanced_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """
        Calculate advanced propagator G^A_{field1,field2}(ω,k).

        Args:
            field1: First field for propagator calculation
            field2: Second field for propagator calculation
            omega_val: Specific frequency value (optional)
            k_val: Specific momentum value (optional)

        Returns:
            Symbolic expression for advanced propagator
        """
        pass

    @abstractmethod
    def calculate_keldysh_propagator(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """
        Calculate Keldysh propagator G^K_{field1,field2}(ω,k).

        Args:
            field1: First field for propagator calculation
            field2: Second field for propagator calculation
            omega_val: Specific frequency value (optional)
            k_val: Specific momentum value (optional)

        Returns:
            Symbolic expression for Keldysh propagator
        """
        pass

    def calculate_spectral_function(
        self,
        field1: Field,
        field2: Field,
        omega_val: complex | None = None,
        k_val: float | None = None,
    ) -> sp.Expr:
        """
        Calculate spectral function A(ω,k) = -2 Im G^R / π.

        Default implementation using retarded propagator.
        Subclasses may override for optimized calculations.
        """
        cache_key = self._create_propagator_cache_key(
            "spectral", field1, field2, omega_val=omega_val, k_val=k_val
        )

        if cache_key in self.spectral_cache:
            # Return cached spectral function expression
            cached_props = self.spectral_cache[cache_key]
            if hasattr(cached_props, "spectral_function"):
                return cached_props.spectral_function

        # Calculate from retarded propagator
        retarded = self.calculate_retarded_propagator(field1, field2, omega_val, k_val)
        spectral = -2 * sp.im(retarded) / sp.pi

        return sp.simplify(spectral)

    def extract_poles(
        self,
        field1: Field,
        field2: Field,
    ) -> SpectralProperties:
        """
        Extract pole structure from propagator.

        Default implementation attempts to solve denominator = 0.
        Subclasses may override for more sophisticated pole finding.
        """
        cache_key = self._create_propagator_cache_key("poles", field1, field2)

        if cache_key in self.spectral_cache:
            return self.spectral_cache[cache_key]

        try:
            retarded = self.calculate_retarded_propagator(field1, field2)

            # Attempt to extract poles by finding zeros of denominator
            # This is a simplified approach - real implementation would be more sophisticated
            poles = sp.solve(sp.denom(retarded), self.omega)

            # Convert to complex numbers if possible
            complex_poles = []
            for pole in poles:
                try:
                    complex_poles.append(complex(pole))
                except (TypeError, ValueError):
                    # Keep as symbolic expression
                    complex_poles.append(pole)

            spectral_props = SpectralProperties(poles=complex_poles)
            spectral_props.validate_causality()

            # Cache result
            self.spectral_cache[cache_key] = spectral_props

            return spectral_props

        except Exception as e:
            warnings.warn(f"Pole extraction failed for {field1.name}-{field2.name}: {str(e)}")
            return SpectralProperties(causality_satisfied=False)

    def verify_fdt_relations(
        self,
        field1: Field,
        field2: Field,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Verify fluctuation-dissipation theorem relations.

        Checks that G^K = (G^R - G^A) coth(ω/2T).
        """
        try:
            retarded = self.calculate_retarded_propagator(field1, field2)
            advanced = self.calculate_advanced_propagator(field1, field2)
            keldysh = self.calculate_keldysh_propagator(field1, field2)

            # FDT relation: G^K = (G^R - G^A) coth(ω/2T)
            expected_keldysh = (retarded - advanced) * sp.coth(self.omega / (2 * self.temperature))

            # Check if they're approximately equal (symbolic comparison is limited)
            difference = sp.simplify(keldysh - expected_keldysh)

            # Basic check - if difference simplifies to zero or small constant
            if difference == 0:
                return True
            elif difference.is_number and abs(complex(difference)) < tolerance:
                return True
            else:
                warnings.warn(
                    f"FDT violation for {field1.name}-{field2.name}: difference = {difference}"
                )
                return False

        except Exception as e:
            warnings.warn(f"FDT verification failed for {field1.name}-{field2.name}: {str(e)}")
            return False

    def construct_propagator_matrix(
        self,
        field_names: list[str],
        matrix_type: str = "retarded",
    ) -> PropagatorMatrix:
        """
        Construct full propagator matrix for given fields.

        Args:
            field_names: List of field names to include in matrix
            matrix_type: Type of propagator matrix ("retarded", "advanced", "keldysh")

        Returns:
            PropagatorMatrix containing the full field-space matrix
        """
        cache_key = f"matrix_{matrix_type}_{'_'.join(field_names)}"

        if cache_key in self.matrix_cache:
            return self.matrix_cache[cache_key]

        # Get fields from action/system
        fields = []
        for name in field_names:
            if hasattr(self.is_system.field_registry, "get_field"):
                field = self.is_system.field_registry.get_field(name)
                if field is not None:
                    fields.append(field)

        if len(fields) != len(field_names):
            warnings.warn(f"Could not find all requested fields: {field_names}")

        # Build matrix
        n = len(fields)
        matrix = sp.zeros(n, n)

        for i in range(n):
            for j in range(n):
                try:
                    if matrix_type == "retarded":
                        element = self.calculate_retarded_propagator(fields[i], fields[j])
                    elif matrix_type == "advanced":
                        element = self.calculate_advanced_propagator(fields[i], fields[j])
                    elif matrix_type == "keldysh":
                        element = self.calculate_keldysh_propagator(fields[i], fields[j])
                    else:
                        raise ValueError(f"Unknown matrix type: {matrix_type}")

                    matrix[i, j] = element

                except Exception as e:
                    warnings.warn(
                        f"Failed to calculate {matrix_type} propagator for {fields[i].name}-{fields[j].name}: {str(e)}"
                    )
                    matrix[i, j] = 0

        prop_matrix = PropagatorMatrix(
            matrix=matrix, field_names=field_names, matrix_type=matrix_type
        )

        # Cache result
        self.matrix_cache[cache_key] = prop_matrix

        return prop_matrix

    def _perform_calculation(self, *args: Any, **kwargs: Any) -> PropagatorComponents:
        """
        Default calculation method that returns all propagator components.

        This provides a unified interface for getting complete propagator information.
        """
        if len(args) >= 2:
            field1, field2 = args[0], args[1]

            # Calculate all components
            retarded = self.calculate_retarded_propagator(field1, field2, **kwargs)
            advanced = self.calculate_advanced_propagator(field1, field2, **kwargs)
            keldysh = self.calculate_keldysh_propagator(field1, field2, **kwargs)
            spectral = self.calculate_spectral_function(field1, field2, **kwargs)

            return PropagatorComponents(
                retarded=retarded, advanced=advanced, keldysh=keldysh, spectral=spectral
            )
        else:
            raise ValueError("PropagatorCalculator requires at least two field arguments")
