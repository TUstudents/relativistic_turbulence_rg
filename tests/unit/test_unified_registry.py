"""
Comprehensive tests for unified field registry architecture.

This module tests the complete unified registry system including:
- AbstractFieldRegistry interface compliance
- Registry conversions between types
- Context-aware registry factory
- Type safety and consistency
- Integration with existing physics calculations

The tests ensure the architectural unification successfully eliminates
type mismatches while maintaining full functionality.
"""

import pytest
import sympy as sp
from sympy import Symbol

from rtrg.core.fields import EnhancedFieldRegistry, Field, FieldRegistry, TensorAwareField
from rtrg.core.registry_base import AbstractFieldRegistry, get_registry_info
from rtrg.core.registry_converter import (
    FieldRegistryConverter,
    convert_core_to_enhanced,
    convert_core_to_symbolic,
    convert_symbolic_to_enhanced,
)
from rtrg.core.registry_factory import (
    RegistryContext,
    RegistryFactory,
    create_registry_for_context,
    get_registry_factory,
)
from rtrg.core.tensors import Metric
from rtrg.field_theory.symbolic_tensors import IndexedFieldRegistry, SymbolicTensorField


class TestAbstractFieldRegistryInterface:
    """Test AbstractFieldRegistry interface and compliance."""

    def test_registry_inheritance_hierarchy(self):
        """Test that all registries properly inherit from AbstractFieldRegistry."""
        # Create instances
        core_registry = FieldRegistry()
        enhanced_registry = EnhancedFieldRegistry()
        symbolic_registry = IndexedFieldRegistry()

        # Check inheritance
        assert isinstance(core_registry, AbstractFieldRegistry)
        assert isinstance(enhanced_registry, AbstractFieldRegistry)
        assert isinstance(symbolic_registry, AbstractFieldRegistry)

        # Check type parameters
        assert core_registry.__class__.__orig_bases__[0].__args__[0] == Field  # type: ignore[attr-defined]
        assert symbolic_registry.__class__.__orig_bases__[0].__args__[0] == SymbolicTensorField  # type: ignore[attr-defined]

    def test_common_interface_methods(self):
        """Test that all registries implement the common interface."""
        registries = [FieldRegistry(), EnhancedFieldRegistry(), IndexedFieldRegistry()]

        for registry in registries:
            # Test basic interface methods exist
            assert hasattr(registry, "get_field")
            assert hasattr(registry, "get_all_fields")
            assert hasattr(registry, "field_count")
            assert hasattr(registry, "list_field_names")
            assert hasattr(registry, "has_field")
            assert hasattr(registry, "is_empty")
            assert hasattr(registry, "registry_type")

            # Test magic methods
            assert hasattr(registry, "__len__")
            assert hasattr(registry, "__contains__")
            assert hasattr(registry, "__str__")
            assert hasattr(registry, "__repr__")

    def test_registry_info_introspection(self):
        """Test registry introspection and info gathering."""
        registry = FieldRegistry()
        registry.create_is_fields()

        info = get_registry_info(registry)

        assert info.registry_type == "FieldRegistry"
        assert info.field_count == 5
        assert set(info.field_names) == {"rho", "u", "pi", "Pi", "q"}
        assert not info.is_empty

    def test_registry_magic_methods(self):
        """Test magic methods work consistently across registries."""
        registry = FieldRegistry()
        registry.create_is_fields()

        # Test __len__
        assert len(registry) == 5

        # Test __contains__
        assert "u" in registry
        assert "nonexistent" not in registry

        # Test __str__ and __repr__
        str_repr = str(registry)
        assert "FieldRegistry" in str_repr
        assert "fields=5" in str_repr

        detailed_repr = repr(registry)
        assert "rho" in detailed_repr
        assert "u" in detailed_repr


class TestFieldRegistryConverter:
    """Test field registry conversion utilities."""

    @pytest.fixture
    def metric(self):
        """Minkowski metric for tests."""
        return Metric()

    @pytest.fixture
    def coordinates(self):
        """Spacetime coordinates for symbolic tests."""
        return [Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    @pytest.fixture
    def converter(self):
        """Field registry converter instance."""
        return FieldRegistryConverter()

    @pytest.fixture
    def core_registry(self, metric):
        """Core field registry with IS fields."""
        registry = FieldRegistry()
        registry.create_is_fields(metric)
        return registry

    def test_core_to_enhanced_conversion(self, converter, core_registry, metric):
        """Test conversion from Core to Enhanced registry."""
        enhanced_registry = converter.core_to_enhanced(core_registry, metric)

        assert isinstance(enhanced_registry, EnhancedFieldRegistry)
        assert enhanced_registry.field_count() == core_registry.field_count()

        # Check field names match
        assert set(enhanced_registry.list_field_names()) == set(core_registry.list_field_names())

        # Check enhanced fields are TensorAware
        for name in enhanced_registry.list_field_names():
            field = enhanced_registry.get_field(name)
            assert isinstance(field, TensorAwareField)

    def test_core_to_symbolic_conversion(self, converter, core_registry, coordinates, metric):
        """Test conversion from Core to Symbolic registry."""
        symbolic_registry = converter.core_to_symbolic(core_registry, coordinates, metric)

        assert isinstance(symbolic_registry, IndexedFieldRegistry)
        assert symbolic_registry.field_count() == core_registry.field_count()

        # Check field names match
        assert set(symbolic_registry.list_field_names()) == set(core_registry.list_field_names())

        # Check symbolic fields are SymbolicTensorField
        for name in symbolic_registry.list_field_names():
            field = symbolic_registry.get_field(name)
            assert isinstance(field, SymbolicTensorField)

    def test_enhanced_to_core_conversion(self, converter, core_registry, metric):
        """Test conversion from Enhanced to Core registry."""
        enhanced_registry = converter.core_to_enhanced(core_registry, metric)
        converted_back = converter.enhanced_to_core(enhanced_registry)

        assert isinstance(converted_back, FieldRegistry)
        assert converted_back.field_count() == enhanced_registry.field_count()

        # Check field names match original
        assert set(converted_back.list_field_names()) == set(core_registry.list_field_names())

    def test_symbolic_to_enhanced_conversion(self, converter, core_registry, coordinates, metric):
        """Test conversion from Symbolic to Enhanced registry."""
        symbolic_registry = converter.core_to_symbolic(core_registry, coordinates, metric)
        enhanced_registry = converter.symbolic_to_enhanced(symbolic_registry, metric)

        assert isinstance(enhanced_registry, EnhancedFieldRegistry)
        assert enhanced_registry.field_count() == symbolic_registry.field_count()

        # Check field names match
        assert set(enhanced_registry.list_field_names()) == set(
            symbolic_registry.list_field_names()
        )

    def test_conversion_caching(self, converter, core_registry, metric):
        """Test that conversions are cached for performance."""
        # Clear cache first
        converter.clear_cache()
        assert converter.cache.size() == 0

        # First conversion should cache result
        enhanced1 = converter.core_to_enhanced(core_registry, metric)
        assert converter.cache.size() == 1

        # Second conversion should use cache
        enhanced2 = converter.core_to_enhanced(core_registry, metric)
        assert enhanced1 is enhanced2  # Same object from cache

    def test_conversion_stats_tracking(self, converter, core_registry, metric, coordinates):
        """Test conversion statistics are tracked correctly."""
        initial_stats = converter.get_conversion_stats()

        # Perform various conversions
        converter.core_to_enhanced(core_registry, metric)
        converter.core_to_symbolic(core_registry, coordinates, metric)

        final_stats = converter.get_conversion_stats()

        # Check stats were updated
        assert final_stats["core_to_enhanced"] > initial_stats["core_to_enhanced"]
        assert final_stats["core_to_symbolic"] > initial_stats["core_to_symbolic"]

    def test_convenience_conversion_functions(self, core_registry, metric, coordinates):
        """Test convenience conversion functions work correctly."""
        # Test convenience functions
        enhanced = convert_core_to_enhanced(core_registry, metric)
        assert isinstance(enhanced, EnhancedFieldRegistry)

        symbolic = convert_core_to_symbolic(core_registry, coordinates, metric)
        assert isinstance(symbolic, IndexedFieldRegistry)

        enhanced_from_symbolic = convert_symbolic_to_enhanced(symbolic, metric)
        assert isinstance(enhanced_from_symbolic, EnhancedFieldRegistry)


class TestRegistryFactory:
    """Test context-aware registry factory."""

    @pytest.fixture
    def factory(self):
        """Registry factory instance."""
        return RegistryFactory()

    @pytest.fixture
    def metric(self):
        """Minkowski metric for tests."""
        return Metric()

    @pytest.fixture
    def coordinates(self):
        """Spacetime coordinates for tests."""
        return [Symbol(name, real=True) for name in ["t", "x", "y", "z"]]

    def test_basic_physics_context(self, factory, metric):
        """Test basic_physics context creates Core FieldRegistry."""
        registry = factory.create_for_context("basic_physics", metric=metric)

        assert isinstance(registry, FieldRegistry)
        assert not isinstance(registry, EnhancedFieldRegistry)  # Should be basic, not enhanced
        assert registry.field_count() == 5
        assert set(registry.list_field_names()) == {"rho", "u", "pi", "Pi", "q"}

    def test_tensor_operations_context(self, factory, metric):
        """Test tensor_operations context creates Enhanced FieldRegistry."""
        registry = factory.create_for_context("tensor_operations", metric=metric)

        assert isinstance(registry, EnhancedFieldRegistry)
        assert registry.field_count() == 5
        assert set(registry.list_field_names()) == {"rho", "u", "pi", "Pi", "q"}

        # Check fields are tensor-aware
        for name in registry.list_field_names():
            field = registry.get_field(name)
            assert isinstance(field, TensorAwareField)

    def test_symbolic_msrjd_context(self, factory, coordinates):
        """Test symbolic_msrjd context creates Indexed FieldRegistry."""
        registry = factory.create_for_context("symbolic_msrjd", coordinates=coordinates)

        assert isinstance(registry, IndexedFieldRegistry)
        assert registry.field_count() == 5
        assert set(registry.list_field_names()) == {"rho", "u", "pi", "Pi", "q"}

        # Check fields are symbolic tensor fields
        for name in registry.list_field_names():
            field = registry.get_field(name)
            assert isinstance(field, SymbolicTensorField)

    def test_registry_context_enum(self, factory, metric):
        """Test RegistryContext enum works with factory."""
        registry = factory.create_for_context(RegistryContext.TENSOR_OPERATIONS, metric=metric)

        assert isinstance(registry, EnhancedFieldRegistry)

    def test_factory_caching(self, factory, metric):
        """Test factory caches registries correctly."""
        factory.clear_cache()

        # Create registry twice with same parameters
        registry1 = factory.create_for_context("tensor_operations", metric=metric)
        registry2 = factory.create_for_context("tensor_operations", metric=metric)

        # Should be same cached instance
        assert registry1 is registry2

    def test_global_factory_functions(self, metric, coordinates):
        """Test global factory convenience functions."""
        # Test global factory access
        factory = get_registry_factory()
        assert isinstance(factory, RegistryFactory)

        # Test global context creation
        registry1 = create_registry_for_context("basic_physics", metric=metric)
        assert isinstance(registry1, FieldRegistry)

        registry2 = create_registry_for_context("symbolic_msrjd", coordinates=coordinates)
        assert isinstance(registry2, IndexedFieldRegistry)

    def test_factory_stats_tracking(self, factory, metric):
        """Test factory tracks creation statistics."""
        initial_stats = factory.get_creation_stats()

        # Create registries of different types
        factory.create_for_context("basic_physics", metric=metric)
        factory.create_for_context("tensor_operations", metric=metric)

        final_stats = factory.get_creation_stats()

        # Check stats were updated
        assert final_stats["basic_physics"] > initial_stats["basic_physics"]
        assert final_stats["tensor_operations"] > initial_stats["tensor_operations"]


class TestUnifiedRegistryIntegration:
    """Test integration of unified registry system with existing code."""

    def test_propagator_matrix_field_compatibility(self):
        """Test PropagatorMatrix works with unified registry fields."""
        from rtrg.field_theory.propagators import PropagatorMatrix

        # Create enhanced registry for tensor operations
        registry = create_registry_for_context("tensor_operations")
        fields = list(registry.get_all_fields().values())[:2]

        # Test PropagatorMatrix accepts these fields
        omega = sp.Symbol("omega", complex=True)
        k_vec = [sp.Symbol(f"k_{i}", real=True) for i in range(3)]
        matrix = sp.Matrix([[1, 2], [3, 4]])

        prop_matrix = PropagatorMatrix(
            matrix=matrix, field_basis=fields, omega=omega, k_vector=k_vec
        )

        # Test field lookup works
        component = prop_matrix.get_component(fields[0], fields[1])
        assert component == 2

    def test_israel_stewart_system_integration(self):
        """Test IsraelStewartSystem works with unified registry."""
        from rtrg.israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem

        # Create IS system (should use unified registry internally)
        params = IsraelStewartParameters()
        system = IsraelStewartSystem(params)

        # Check it has proper registry with unified interface
        assert isinstance(system.field_registry, AbstractFieldRegistry)
        assert system.field_registry.field_count() == 5
        assert set(system.field_registry.list_field_names()) == {"rho", "u", "pi", "Pi", "q"}

    def test_registry_conversion_integration(self):
        """Test registry conversions work in realistic scenarios."""
        from rtrg.core.tensors import Metric

        # Start with basic registry (like IsraelStewartSystem creates)
        basic_registry = create_registry_for_context("basic_physics")

        # Convert to enhanced for advanced tensor operations
        enhanced_registry = convert_core_to_enhanced(basic_registry, Metric())

        # Convert to symbolic for MSRJD calculations
        coordinates = [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]
        symbolic_registry = convert_core_to_symbolic(basic_registry, coordinates, Metric())

        # All should have same field structure
        assert (
            basic_registry.field_count()
            == enhanced_registry.field_count()
            == symbolic_registry.field_count()
        )
        assert (
            set(basic_registry.list_field_names())
            == set(enhanced_registry.list_field_names())
            == set(symbolic_registry.list_field_names())
        )

    def test_type_safety_across_registries(self):
        """Test type safety is maintained across unified registry system."""
        # Create registries of different types
        basic_reg = create_registry_for_context("basic_physics")
        enhanced_reg = create_registry_for_context("tensor_operations")
        coordinates = [sp.Symbol(name, real=True) for name in ["t", "x", "y", "z"]]
        symbolic_reg = create_registry_for_context("symbolic_msrjd", coordinates=coordinates)

        # Test fields have correct types
        basic_field = basic_reg.get_field("u")
        enhanced_field = enhanced_reg.get_field("u")
        symbolic_field = symbolic_reg.get_field("u")

        assert isinstance(basic_field, Field)
        assert isinstance(enhanced_field, TensorAwareField)
        assert isinstance(symbolic_field, SymbolicTensorField)

        # Test registry types
        assert isinstance(basic_reg, FieldRegistry)
        assert isinstance(enhanced_reg, EnhancedFieldRegistry)
        assert isinstance(symbolic_reg, IndexedFieldRegistry)

    def test_no_more_manual_conversions(self):
        """Test that manual registry conversions are eliminated."""
        # Before unification, code had manual conversions like:
        # enhanced_registry = EnhancedFieldRegistry()
        # enhanced_registry.create_enhanced_is_fields(metric)

        # After unification, context-aware creation eliminates this:
        registry = create_registry_for_context("tensor_operations", metric=Metric())

        assert isinstance(registry, EnhancedFieldRegistry)
        assert registry.field_count() == 5
        assert "u" in registry

        # No more manual field creation needed - factory handles everything

    def test_backward_compatibility_maintained(self):
        """Test that existing code patterns still work."""
        # Old patterns should still work alongside new unified system
        old_style_registry = FieldRegistry()
        old_style_registry.create_is_fields()

        new_style_registry = create_registry_for_context("basic_physics")

        # Both should produce equivalent results
        assert old_style_registry.field_count() == new_style_registry.field_count()
        assert set(old_style_registry.list_field_names()) == set(
            new_style_registry.list_field_names()
        )

        # Both should work with existing code
        assert len(old_style_registry) == len(new_style_registry)
        assert "u" in old_style_registry and "u" in new_style_registry
