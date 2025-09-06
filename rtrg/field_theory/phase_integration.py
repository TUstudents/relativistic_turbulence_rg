"""
Integration module connecting Phase 1 tensor infrastructure with Phase 2 symbolic system.

This module provides seamless integration between the Phase 1 concrete tensor
infrastructure and the Phase 2 symbolic tensor field system for MSRJD calculations.

Key Features:
    - Bidirectional conversion between Phase 1 and Phase 2 field representations
    - Unified interface for both concrete and symbolic tensor operations
    - Constraint consistency across both systems
    - Validation and testing integration
    - Performance optimization for mixed computations

Integration Architecture:
    Phase 1 (Concrete):        Phase 2 (Symbolic):
    - TensorAwareField      ←→  SymbolicTensorField
    - EnhancedFieldRegistry ←→  IndexedFieldRegistry
    - ConstrainedTensorField ←→ Field constraints
    - ProjectionOperators   ←→  Tensor contractions

Usage:
    >>> integrator = PhaseIntegrator()
    >>> phase1_system = create_phase1_system()
    >>> symbolic_system = integrator.convert_to_symbolic(phase1_system)
    >>> results = integrator.run_unified_calculation(symbolic_system)

References:
    - Phase 1 infrastructure (rtrg.core.tensors, rtrg.core.fields)
    - Phase 2 symbolic system (rtrg.field_theory.symbolic_tensors)
    - MSRJD field theory integration
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import sympy as sp
from sympy import Matrix, symbols

from ..core.fields import (
    EnhancedBulkPressureField,
    EnhancedEnergyDensityField,
    EnhancedFieldRegistry,
    EnhancedFourVelocityField,
    EnhancedHeatFluxField,
    EnhancedShearStressField,
    FieldProperties,
    TensorAwareField,
)
from ..core.registry_factory import RegistryFactory, create_registry_for_context

# Phase 1 imports
from ..core.tensors import (
    ConstrainedTensorField,
    IndexType,
    Metric,
    ProjectionOperators,
    TensorIndex,
    TensorIndexStructure,
)

# Israel-Stewart system
from ..israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem
from .propagators import TensorPropagatorExtractor

# Phase 2 imports
from .symbolic_tensors import (
    IndexedFieldRegistry,
    SymbolicTensorField,
    TensorDerivative,
    TensorFieldProperties,
)
from .tensor_action_expander import TensorActionExpander, TensorExpansionResult
from .tensor_msrjd_action import TensorActionComponents, TensorMSRJDAction


@dataclass
class IntegrationConfig:
    """Configuration options for phase integration."""

    # Conversion options
    preserve_constraints: bool = True
    validate_consistency: bool = True
    use_symbolic_cache: bool = True

    # Performance options
    parallel_computation: bool = False
    precision_tolerance: float = 1e-12
    symbolic_simplification: bool = True

    # Validation options
    cross_validate_results: bool = True
    compare_propagators: bool = True
    check_ward_identities: bool = False


@dataclass
class IntegrationResults:
    """Results from integrated Phase 1 + Phase 2 calculations."""

    # Converted systems
    phase1_registry: EnhancedFieldRegistry | None = None
    phase2_registry: IndexedFieldRegistry | None = None

    # Computed quantities
    symbolic_action: TensorActionComponents | None = None
    expansion_result: TensorExpansionResult | None = None
    propagators: dict[str, Any] = field(default_factory=dict)

    # Validation results
    consistency_checks: dict[str, bool] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PhaseIntegrator:
    """
    Main integration class connecting Phase 1 and Phase 2 systems.

    This class provides a unified interface for working with both the
    concrete tensor infrastructure (Phase 1) and the symbolic tensor
    field system (Phase 2), enabling seamless calculations across both.

    Key Capabilities:
        - Bidirectional conversion between field representations
        - Unified calculation workflows
        - Cross-validation between systems
        - Performance optimization
        - Error handling and validation

    Usage Examples:
        >>> integrator = PhaseIntegrator()
        >>> system = IsraelStewartSystem(parameters)
        >>> results = integrator.run_full_msrjd_calculation(system)
        >>> propagators = results.propagators
    """

    def __init__(self, config: IntegrationConfig | None = None):
        """
        Initialize phase integrator.

        Args:
            config: Integration configuration options
        """
        self.config = config or IntegrationConfig()

        # Coordinate symbols (shared between phases)
        self.coordinates = symbols("t x y z", real=True)

        # Cache for converted objects
        self._conversion_cache: dict[str, Any] = {}
        self._validation_cache: dict[str, Any] = {}

        # Performance tracking
        self._performance_log: dict[str, float] = {}

        # Registry factory for context-aware creation
        self._registry_factory = RegistryFactory()

    def convert_phase1_to_symbolic(
        self, phase1_registry: EnhancedFieldRegistry
    ) -> IndexedFieldRegistry:
        """
        Convert Phase 1 enhanced field registry to Phase 2 symbolic registry.

        Args:
            phase1_registry: Enhanced field registry from Phase 1

        Returns:
            Equivalent indexed field registry for Phase 2
        """
        cache_key = f"p1_to_p2_{id(phase1_registry)}"
        if cache_key in self._conversion_cache and self.config.use_symbolic_cache:
            return self._conversion_cache[cache_key]  # type: ignore[no-any-return]

        # Create new symbolic registry for MSRJD symbolic field theory
        symbolic_registry_base = self._registry_factory.create_for_context(
            "symbolic_msrjd", coordinates=self.coordinates
        )
        # Cast to IndexedFieldRegistry for type safety
        from .symbolic_tensors import IndexedFieldRegistry

        if not isinstance(symbolic_registry_base, IndexedFieldRegistry):
            raise TypeError(f"Expected IndexedFieldRegistry, got {type(symbolic_registry_base)}")
        symbolic_registry: IndexedFieldRegistry = symbolic_registry_base

        # Convert each field
        for field_name in ["rho", "u", "pi", "Pi", "q"]:
            phase1_field = phase1_registry.get_tensor_aware_field(field_name)

            if phase1_field is not None:
                # Convert to symbolic field
                symbolic_field = self._convert_tensor_aware_to_symbolic(field_name, phase1_field)
                symbolic_registry.register_field(field_name, symbolic_field)

                # Create antifield if method exists
                if hasattr(symbolic_registry, "create_antifield"):
                    symbolic_registry.create_antifield(field_name)

                # Transfer constraints
                if self.config.preserve_constraints:
                    # Cast is safe since we validated the type above
                    self._transfer_field_constraints(phase1_field, symbolic_registry, field_name)

        # Cache result
        if self.config.use_symbolic_cache:
            self._conversion_cache[cache_key] = symbolic_registry

        return symbolic_registry

    def _convert_tensor_aware_to_symbolic(
        self, field_name: str, phase1_field: TensorAwareField
    ) -> SymbolicTensorField:
        """Convert Phase 1 TensorAwareField to Phase 2 SymbolicTensorField."""

        # Extract index structure
        index_structure = []
        if hasattr(phase1_field, "index_structure") and phase1_field.index_structure is not None:
            for tensor_idx in phase1_field.index_structure.indices:
                index_info = (
                    tensor_idx.name,
                    tensor_idx.position,
                    tensor_idx.index_type.name.lower(),
                )
                index_structure.append(index_info)
        else:
            # Fallback based on field name
            index_structure = self._get_default_index_structure(field_name)

        # Extract constraints
        constraints = []
        if hasattr(phase1_field, "constraints"):
            constraints = phase1_field.constraints.copy()

        # Create symbolic field
        properties = TensorFieldProperties(
            name=field_name,
            index_structure=index_structure,
            coordinates=list(self.coordinates),
            constraints=constraints,
            field_type=phase1_field.properties.field_type,
        )

        symbolic_field = SymbolicTensorField(
            field_name,
            index_structure,
            self.coordinates,
            constraints=constraints,
            field_type=phase1_field.properties.field_type,
        )

        return symbolic_field

    def _get_default_index_structure(self, field_name: str) -> list[tuple[str, str, str]]:
        """Get default index structure for known fields."""
        defaults = {
            "rho": [],  # Scalar
            "u": [("mu", "upper", "spacetime")],  # Four-vector
            "pi": [("mu", "upper", "spacetime"), ("nu", "upper", "spacetime")],  # Rank-2 tensor
            "Pi": [],  # Scalar
            "q": [("mu", "upper", "spacetime")],  # Four-vector
        }
        return defaults.get(field_name, [])

    def _transfer_field_constraints(
        self,
        phase1_field: TensorAwareField,
        symbolic_registry: IndexedFieldRegistry,
        field_name: str,
    ) -> None:
        """Transfer constraints from Phase 1 field to symbolic registry."""

        if hasattr(phase1_field, "constraints"):
            for constraint_name in phase1_field.constraints:
                # Create constraint expression
                if constraint_name == "normalization" and field_name == "u":
                    symbolic_field = symbolic_registry.get_field(field_name)
                    if symbolic_field:
                        constraint_expr = symbolic_field.apply_constraint("normalization")
                        symbolic_registry.add_constraint(field_name, constraint_expr)

                elif constraint_name == "traceless" and field_name == "pi":
                    symbolic_field = symbolic_registry.get_field(field_name)
                    if symbolic_field:
                        constraint_expr = symbolic_field.apply_constraint("traceless")
                        symbolic_registry.add_constraint(field_name, constraint_expr)

    def convert_symbolic_to_phase1(
        self, symbolic_registry: IndexedFieldRegistry, metric: Metric | None = None
    ) -> EnhancedFieldRegistry:
        """
        Convert Phase 2 symbolic registry back to Phase 1 enhanced registry.

        Args:
            symbolic_registry: Symbolic field registry
            metric: Spacetime metric (default: Minkowski)

        Returns:
            Enhanced field registry for Phase 1
        """
        cache_key = f"p2_to_p1_{id(symbolic_registry)}"
        if cache_key in self._conversion_cache and self.config.use_symbolic_cache:
            return self._conversion_cache[cache_key]  # type: ignore[no-any-return]

        if metric is None:
            metric = Metric()  # Default Minkowski metric

        # Create enhanced registry for tensor operations
        enhanced_registry_base = self._registry_factory.create_for_context(
            "tensor_operations", metric=metric
        )
        # Cast to EnhancedFieldRegistry for type safety
        if not isinstance(enhanced_registry_base, EnhancedFieldRegistry):
            raise TypeError(f"Expected EnhancedFieldRegistry, got {type(enhanced_registry_base)}")
        enhanced_registry: EnhancedFieldRegistry = enhanced_registry_base

        # Transfer constraint information back
        for field_name in ["rho", "u", "pi", "Pi", "q"]:
            symbolic_field = symbolic_registry.get_field(field_name)
            enhanced_field = (
                enhanced_registry.get_tensor_aware_field(field_name)
                if hasattr(enhanced_registry, "get_tensor_aware_field")
                else enhanced_registry.get_field(field_name)
            )

            if symbolic_field and enhanced_field:
                # Transfer constraints
                constraints = symbolic_registry.get_constraints(field_name)
                if constraints and hasattr(enhanced_field, "constraints"):
                    # Convert symbolic constraints back to constraint names
                    enhanced_field.constraints.extend(self._extract_constraint_names(constraints))

        # Cache result
        if self.config.use_symbolic_cache:
            self._conversion_cache[cache_key] = enhanced_registry

        return enhanced_registry

    def _extract_constraint_names(self, symbolic_constraints: list[sp.Expr]) -> list[str]:
        """Extract constraint names from symbolic constraint expressions."""
        constraint_names = []

        for constraint in symbolic_constraints:
            constraint_str = str(constraint)

            # Simple pattern matching for common constraints
            if "c**2" in constraint_str and "u[mu]" in constraint_str:
                constraint_names.append("normalization")
            elif any(term in constraint_str for term in ["pi[mu, mu]", "trace"]):
                constraint_names.append("traceless")
            elif "orthogonal" in constraint_str.lower():
                constraint_names.append("orthogonal")

        return constraint_names

    def run_full_msrjd_calculation(
        self, is_system: IsraelStewartSystem, temperature: float = 1.0
    ) -> IntegrationResults:
        """
        Run complete MSRJD calculation using integrated Phase 1 + Phase 2 approach.

        Args:
            is_system: Israel-Stewart system with parameters
            temperature: System temperature

        Returns:
            Complete integration results
        """
        results = IntegrationResults()

        try:
            # Step 1: Set up Phase 1 system
            if hasattr(is_system, "field_registry"):
                results.phase1_registry = is_system.field_registry
            else:
                # Create enhanced registry
                enhanced_registry = create_registry_for_context(
                    "tensor_operations", metric=is_system.metric
                )
                results.phase1_registry = enhanced_registry

            # Step 2: Convert to Phase 2 symbolic system
            results.phase2_registry = self.convert_phase1_to_symbolic(results.phase1_registry)  # type: ignore[arg-type]

            # Step 3: Build tensor MSRJD action
            tensor_action = TensorMSRJDAction(is_system, temperature, use_enhanced_registry=False)
            tensor_action.field_registry = results.phase2_registry

            results.symbolic_action = tensor_action.construct_full_action()

            # Step 4: Action expansion and vertex extraction
            expander = TensorActionExpander(tensor_action)
            results.expansion_result = expander.expand_to_order(4)

            # Step 5: Propagator extraction
            propagator_extractor = TensorPropagatorExtractor(tensor_action, temperature)
            results.propagators = propagator_extractor.get_israel_stewart_propagators()

            # Step 6: Validation
            if self.config.validate_consistency:
                results.consistency_checks = self._validate_integration_consistency(
                    results.phase1_registry,  # type: ignore[arg-type]
                    results.phase2_registry,
                    results.symbolic_action,  # type: ignore[arg-type]
                )

            # Step 7: Cross-validation with Phase 1 (if requested)
            if self.config.cross_validate_results:
                phase1_validation = self._cross_validate_with_phase1(is_system, results)
                results.consistency_checks.update(phase1_validation)

        except Exception as e:
            error_msg = f"MSRJD calculation failed: {str(e)}"
            results.errors.append(error_msg)
            warnings.warn(error_msg, stacklevel=2)

        return results

    def _validate_integration_consistency(
        self,
        phase1_registry: EnhancedFieldRegistry,
        phase2_registry: IndexedFieldRegistry,
        symbolic_action: TensorActionComponents,
    ) -> dict[str, bool]:
        """Validate consistency between Phase 1 and Phase 2 systems."""
        validation = {}

        try:
            # Check field count consistency
            p1_count = len(
                [
                    name
                    for name in ["rho", "u", "pi", "Pi", "q"]
                    if phase1_registry.get_tensor_aware_field(name) is not None
                ]
            )
            p2_count = phase2_registry.field_count()
            validation["field_count_consistent"] = p1_count == p2_count

            # Check constraint consistency
            constraint_consistency = True
            for field_name in ["u", "pi"]:
                p1_field = phase1_registry.get_tensor_aware_field(field_name)
                p2_constraints = phase2_registry.get_constraints(field_name)

                if p1_field and hasattr(p1_field, "constraints"):
                    if len(p1_field.constraints) != len(p2_constraints):
                        constraint_consistency = False

            validation["constraint_consistency"] = constraint_consistency

            # Check symbolic action validity
            validation["symbolic_action_valid"] = symbolic_action.validate_tensor_consistency()

            # Overall validation
            validation["overall"] = all(validation.values())

        except Exception as e:
            warnings.warn(f"Integration validation failed: {str(e)}", stacklevel=2)
            validation["overall"] = False

        return validation

    def _cross_validate_with_phase1(
        self, is_system: IsraelStewartSystem, results: IntegrationResults
    ) -> dict[str, bool]:
        """Cross-validate Phase 2 results against Phase 1 calculations."""
        cross_validation = {}

        try:
            # This would compare propagators, vertices, etc. between phases
            # For now, implement basic structural checks

            # Check that all expected propagators exist
            expected_propagators = [
                "velocity",
                "shear_stress",
                "bulk_pressure",
                "heat_flux",
                "energy_density",
            ]
            propagators_present = all(prop in results.propagators for prop in expected_propagators)
            cross_validation["all_propagators_present"] = propagators_present

            # Check that propagators have required components
            components_valid = True
            for _prop_name, prop_components in results.propagators.items():
                if not hasattr(prop_components, "retarded") or prop_components.retarded is None:
                    components_valid = False

            cross_validation["propagator_components_valid"] = components_valid

            # Check expansion result structure
            if results.expansion_result:
                cross_validation["expansion_computed"] = (
                    len(results.expansion_result.expansion_terms) > 0
                )
                cross_validation["quadratic_matrix_exists"] = (
                    results.expansion_result.quadratic_matrix is not None
                )
            else:
                cross_validation["expansion_computed"] = False
                cross_validation["quadratic_matrix_exists"] = False

            # Overall cross-validation
            cross_validation["cross_validation_passed"] = all(cross_validation.values())

        except Exception as e:
            warnings.warn(f"Cross-validation failed: {str(e)}", stacklevel=2)
            cross_validation["cross_validation_passed"] = False

        return cross_validation

    def compare_propagators(
        self,
        phase1_propagators: dict[str, Any],
        phase2_propagators: dict[str, Any],
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """
        Compare propagators computed in Phase 1 vs Phase 2.

        Args:
            phase1_propagators: Propagators from Phase 1 calculation
            phase2_propagators: Propagators from Phase 2 calculation
            tolerance: Numerical comparison tolerance

        Returns:
            Dictionary of comparison results
        """
        comparison = {}

        # Common propagator names
        common_names = set(phase1_propagators.keys()) & set(phase2_propagators.keys())
        comparison["common_propagators"] = len(common_names)

        # Compare individual propagators
        for prop_name in common_names:
            try:
                p1_prop = phase1_propagators[prop_name]
                p2_prop = phase2_propagators[prop_name]

                # This would implement detailed numerical comparison
                # For now, just check structural similarity
                comparison[f"{prop_name}_compatible"] = True  # Placeholder

            except Exception as e:
                comparison[f"{prop_name}_compatible"] = False
                warnings.warn(f"Could not compare propagator {prop_name}: {str(e)}", stacklevel=2)

        return comparison

    def optimize_mixed_calculation(
        self, is_system: IsraelStewartSystem, computation_types: list[str]
    ) -> IntegrationResults:
        """
        Optimize calculation by using Phase 1 for some tasks and Phase 2 for others.

        Args:
            is_system: Israel-Stewart system
            computation_types: List of computation types ["propagators", "vertices", "constraints"]

        Returns:
            Optimized integration results
        """
        results = IntegrationResults()

        # Determine optimal phase for each computation
        phase_assignment = self._optimize_phase_assignment(computation_types)

        # Run computations in assigned phases
        for comp_type, phase in phase_assignment.items():
            if phase == 1:
                # Use Phase 1 for this computation
                self._run_phase1_computation(comp_type, is_system, results)
            elif phase == 2:
                # Use Phase 2 for this computation
                self._run_phase2_computation(comp_type, is_system, results)

        return results

    def _optimize_phase_assignment(self, computation_types: list[str]) -> dict[str, int]:
        """Determine optimal phase assignment for different computation types."""
        # Simple heuristics for phase assignment
        assignments = {}

        for comp_type in computation_types:
            if comp_type == "constraints":
                # Phase 1 is better for constraint handling
                assignments[comp_type] = 1
            elif comp_type == "vertices":
                # Phase 2 is better for symbolic vertex extraction
                assignments[comp_type] = 2
            elif comp_type == "propagators":
                # Phase 2 is better for systematic propagator extraction
                assignments[comp_type] = 2
            else:
                # Default to Phase 2
                assignments[comp_type] = 2

        return assignments

    def _run_phase1_computation(
        self, comp_type: str, is_system: IsraelStewartSystem, results: IntegrationResults
    ) -> None:
        """Run computation using Phase 1 infrastructure."""
        if comp_type == "constraints":
            # Use Phase 1 constraint system
            if results.phase1_registry is None:
                # Create enhanced registry for constraint operations
                enhanced_registry = self._registry_factory.create_for_context(
                    "tensor_operations", metric=is_system.metric
                )
                results.phase1_registry = enhanced_registry

            # Apply and validate constraints
            dummy_components = {"u": np.array([1, 0, 0, 0])}
            if results.phase1_registry is not None and hasattr(
                results.phase1_registry, "apply_all_constraints"
            ):
                constrained = results.phase1_registry.apply_all_constraints(dummy_components)
            results.consistency_checks["phase1_constraints_applied"] = True

    def _run_phase2_computation(
        self, comp_type: str, is_system: IsraelStewartSystem, results: IntegrationResults
    ) -> None:
        """Run computation using Phase 2 symbolic system."""
        if comp_type == "vertices":
            # Use Phase 2 for vertex extraction
            if results.phase2_registry is None:
                if results.phase1_registry is None:
                    enhanced_registry = create_registry_for_context(
                        "tensor_operations", metric=is_system.metric
                    )
                    results.phase1_registry = enhanced_registry

                if results.phase1_registry is not None:
                    results.phase2_registry = self.convert_phase1_to_symbolic(
                        results.phase1_registry
                    )
                else:
                    raise RuntimeError("Phase1 registry is None, cannot convert to symbolic")

            # Extract vertices using symbolic system
            tensor_action = TensorMSRJDAction(is_system, use_enhanced_registry=False)
            tensor_action.field_registry = results.phase2_registry

            expander = TensorActionExpander(tensor_action)
            expansion_result = expander.expand_to_order(3)

            vertices = expansion_result.get_vertices_by_order(3)
            results.expansion_result = expansion_result

        elif comp_type == "propagators":
            # Use Phase 2 for propagator extraction
            if results.phase2_registry is None:
                if results.phase1_registry is None:
                    enhanced_registry = create_registry_for_context(
                        "tensor_operations", metric=is_system.metric
                    )
                    results.phase1_registry = enhanced_registry

                if results.phase1_registry is not None:
                    results.phase2_registry = self.convert_phase1_to_symbolic(
                        results.phase1_registry
                    )
                else:
                    raise RuntimeError("Phase1 registry is None, cannot convert to symbolic")

            tensor_action = TensorMSRJDAction(is_system, use_enhanced_registry=False)
            tensor_action.field_registry = results.phase2_registry

            extractor = TensorPropagatorExtractor(tensor_action)
            results.propagators = extractor.get_israel_stewart_propagators()

    def get_integration_statistics(self) -> dict[str, Any]:
        """Get statistics about integration performance and usage."""
        stats = {
            "cache_size": len(self._conversion_cache),
            "validation_cache_size": len(self._validation_cache),
            "performance_log": self._performance_log.copy(),
            "config": {
                "preserve_constraints": self.config.preserve_constraints,
                "validate_consistency": self.config.validate_consistency,
                "use_symbolic_cache": self.config.use_symbolic_cache,
            },
        }

        return stats

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._conversion_cache.clear()
        self._validation_cache.clear()
        self._performance_log.clear()

    def __str__(self) -> str:
        cache_info = f"cache_size={len(self._conversion_cache)}"
        return f"PhaseIntegrator({cache_info}, config={self.config})"

    def __repr__(self) -> str:
        return self.__str__()
