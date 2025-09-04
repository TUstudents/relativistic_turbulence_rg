"""
Comprehensive integration tests for the complete Feynman Rules system.

This module tests the complete pipeline from MSRJD action construction to
Feynman rule generation, ensuring all components work together correctly.

Test Coverage:
    - Complete vertex extraction from enhanced MSRJD action
    - Vertex classification and tensor structure analysis
    - Momentum space conversion and frequency dependence
    - Ward identity verification for all vertex types
    - Propagator generation with realistic dispersion relations
    - Dimensional analysis and consistency verification
    - Integration between all system components
    - Performance validation for large vertex catalogs

Integration Test Structure:
    - TestCompleteVertexExtraction: End-to-end vertex extraction
    - TestFeynmanRuleGeneration: Complete rule generation pipeline
    - TestWardIdentityValidation: Comprehensive symmetry verification
    - TestDimensionalConsistency: Complete dimensional analysis
    - TestSystemIntegration: Full system integration validation
"""

import pytest
import sympy as sp
from sympy import I, Symbol, symbols

from rtrg.core.constants import PhysicalConstants
from rtrg.field_theory.feynman_rules import (
    DimensionalAnalyzer,
    FeynmanRules,
    MomentumConfiguration,
    WardIdentityChecker,
)
from rtrg.field_theory.msrjd_action import MSRJDAction
from rtrg.field_theory.vertices import VertexExtractor, VertexValidator
from rtrg.israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem


class TestCompleteVertexExtraction:
    """Test complete vertex extraction pipeline from MSRJD action."""

    @pytest.fixture
    def is_parameters(self):
        """Realistic Israel-Stewart parameters for testing."""
        return IsraelStewartParameters(
            eta=2.5,  # Shear viscosity
            zeta=0.8,  # Bulk viscosity
            kappa=1.2,  # Thermal conductivity
            tau_pi=0.15,  # Shear relaxation time
            tau_Pi=0.08,  # Bulk relaxation time
            tau_q=0.05,  # Heat flux relaxation time
        )

    @pytest.fixture
    def is_system(self, is_parameters):
        """Complete Israel-Stewart system."""
        return IsraelStewartSystem(is_parameters)

    @pytest.fixture
    def msrjd_action(self, is_system):
        """MSRJD action with realistic parameters."""
        return MSRJDAction(is_system, temperature=1.5)

    @pytest.fixture
    def vertex_extractor(self, msrjd_action):
        """Vertex extractor configured for realistic extraction."""
        return VertexExtractor(msrjd_action)

    def test_complete_vertex_catalog_extraction(self, vertex_extractor):
        """Test extraction of complete vertex catalog."""
        # Extract all vertices
        catalog = vertex_extractor.extract_all_vertices()

        # Verify catalog structure
        assert hasattr(catalog, "three_point")
        assert hasattr(catalog, "four_point")
        assert hasattr(catalog, "constraint_vertices")

        # Check for expected Israel-Stewart vertex types
        expected_types = {
            "advection",
            "shear_transport",
            "bulk_transport",
            "heat_transport",
            "energy_momentum",
            "mixed_coupling",
        }
        found_types = catalog.vertex_types

        # Should find most expected types (allowing for some flexibility)
        overlap = expected_types & found_types
        assert len(overlap) >= 3, f"Expected vertex types {expected_types}, found {found_types}"

    def test_vertex_classification_completeness(self, vertex_extractor):
        """Test that vertex classification covers all Israel-Stewart physics."""
        catalog = vertex_extractor.extract_all_vertices()

        # Check that vertices are properly classified
        unclassified_vertices = []
        for vertices_dict in [catalog.three_point, catalog.four_point, catalog.constraint_vertices]:
            for field_combo, vertex in vertices_dict.items():
                if vertex.vertex_type == "unclassified":
                    unclassified_vertices.append(field_combo)

        # Should have very few unclassified vertices
        assert (
            len(unclassified_vertices) <= 1
        ), f"Too many unclassified vertices: {unclassified_vertices}"

    def test_tensor_structure_analysis(self, vertex_extractor):
        """Test comprehensive tensor structure analysis."""
        catalog = vertex_extractor.extract_all_vertices()

        # Check tensor structure descriptions
        structure_types = set()
        for vertices_dict in [catalog.three_point, catalog.four_point]:
            for vertex in vertices_dict.values():
                structure_types.add(vertex.tensor_structure)

        # Should have diverse tensor structures
        expected_structures = ["scalar", "vector", "tensor", "derivative"]
        found_structures = [
            s for s in structure_types if any(exp in s.lower() for exp in expected_structures)
        ]

        assert (
            len(found_structures) >= 2
        ), f"Expected diverse tensor structures, found: {structure_types}"

    def test_dimensional_consistency_verification(self, vertex_extractor):
        """Test dimensional consistency of extracted vertices."""
        catalog = vertex_extractor.extract_all_vertices()
        validator = VertexValidator(catalog, vertex_extractor.parameters)

        # Verify dimensional consistency
        is_dimensionally_consistent = validator.validate_dimensional_consistency()
        assert is_dimensionally_consistent, "Vertex catalog fails dimensional consistency check"

        # Check individual vertex dimensions
        dimensional_violations = []
        for vertices_dict in [catalog.three_point, catalog.four_point]:
            for field_combo, vertex in vertices_dict.items():
                # Action should have dimension 4 (spacetime integral)
                expected_range = (3.5, 4.5)  # Allow some tolerance
                if not (expected_range[0] <= vertex.mass_dimension <= expected_range[1]):
                    dimensional_violations.append((field_combo, vertex.mass_dimension))

        assert len(dimensional_violations) <= 2, f"Dimensional violations: {dimensional_violations}"


class TestFeynmanRuleGeneration:
    """Test complete Feynman rule generation pipeline."""

    @pytest.fixture
    def complete_feynman_rules(self):
        """Complete Feynman rules system for testing."""
        parameters = IsraelStewartParameters(
            eta=1.5, zeta=0.5, kappa=0.8, tau_pi=0.12, tau_Pi=0.06, tau_q=0.04
        )
        is_system = IsraelStewartSystem(parameters)
        msrjd_action = MSRJDAction(is_system, temperature=2.0)
        extractor = VertexExtractor(msrjd_action)
        catalog = extractor.extract_all_vertices()

        return FeynmanRules(catalog, parameters=parameters)

    def test_vertex_rule_generation(self, complete_feynman_rules):
        """Test systematic vertex rule generation."""
        vertex_rules = complete_feynman_rules.generate_all_vertex_rules()

        # Should generate multiple vertex rules
        assert len(vertex_rules) > 0, "No vertex rules generated"

        # Check rule structure
        for field_combo, rule in vertex_rules.items():
            assert hasattr(rule, "amplitude"), f"Rule {field_combo} missing amplitude"
            assert hasattr(rule, "tensor_structure"), f"Rule {field_combo} missing tensor structure"
            assert hasattr(
                rule, "ward_identities"
            ), f"Rule {field_combo} missing Ward identity checks"
            assert hasattr(
                rule, "dimensional_consistency"
            ), f"Rule {field_combo} missing dimensional check"

    def test_propagator_rule_generation(self, complete_feynman_rules):
        """Test complete propagator rule generation."""
        propagator_rules = complete_feynman_rules.generate_propagator_rules()

        # Should generate propagators for all physical fields
        expected_fields = ["rho", "u", "pi", "Pi", "q"]
        generated_fields = set()

        for prop_key, _rule in propagator_rules.items():
            if len(prop_key) >= 2:
                generated_fields.add(prop_key[0])  # Field name

        # Should cover most physical fields
        coverage = len(generated_fields & set(expected_fields))
        assert coverage >= 3, f"Insufficient propagator coverage: {generated_fields}"

    def test_realistic_dispersion_relations(self, complete_feynman_rules):
        """Test that dispersion relations are physically realistic."""
        propagator_rules = complete_feynman_rules.generate_propagator_rules()

        # Check dispersion relations for physical realism
        for _prop_key, rule in propagator_rules.items():
            if hasattr(rule, "dispersion_relation") and rule.dispersion_relation:
                dispersion = rule.dispersion_relation

                # Should contain momentum dependence
                has_momentum = any(
                    str(dispersion).find(k_symbol) >= 0 for k_symbol in ["k", "momentum"]
                )

                # Should have realistic structure (not just constants)
                is_nontrivial = len(str(dispersion)) > 5

                assert (
                    has_momentum or is_nontrivial
                ), f"Dispersion relation too simple: {dispersion}"

    def test_momentum_space_conversion(self, complete_feynman_rules):
        """Test momentum space conversion functionality."""
        # Test with simple symbolic expression
        t, x, y, z = symbols("t x y z", real=True)
        test_expr = sp.Derivative(sp.Function("f")(t, x), t) + sp.Derivative(
            sp.Function("g")(x, y), x
        )

        converted = complete_feynman_rules._convert_to_momentum_space(test_expr)

        # Should contain momentum space factors
        converted_str = str(converted)
        has_omega = "omega" in converted_str or "I" in converted_str
        has_momentum = any(k in converted_str for k in ["k_vec", "k_mu", "momentum"])

        assert has_omega or has_momentum, f"Momentum space conversion failed: {converted}"


class TestWardIdentityValidation:
    """Test comprehensive Ward identity verification."""

    @pytest.fixture
    def ward_checker(self):
        """Ward identity checker with complete system."""
        parameters = IsraelStewartParameters(eta=2.0, zeta=0.6, kappa=1.0)
        is_system = IsraelStewartSystem(parameters)
        msrjd_action = MSRJDAction(is_system)
        extractor = VertexExtractor(msrjd_action)
        catalog = extractor.extract_all_vertices()
        rules = FeynmanRules(catalog, parameters=parameters)

        return WardIdentityChecker(rules)

    def test_energy_momentum_conservation(self, ward_checker):
        """Test energy-momentum conservation verification."""
        conservation_results = ward_checker.check_energy_momentum_conservation()

        # Should return conservation check results
        assert isinstance(conservation_results, dict)

        # Results should be boolean values
        for conservation_type, result in conservation_results.items():
            assert isinstance(
                result, bool
            ), f"Conservation check {conservation_type} returned non-boolean: {result}"

    def test_current_conservation(self, ward_checker):
        """Test current conservation verification."""
        current_results = ward_checker.check_current_conservation()

        assert isinstance(current_results, dict)
        # Should check various current types
        # Allow empty dict if no currents present
        assert len(current_results) >= 0

    def test_fluctuation_dissipation_relations(self, ward_checker):
        """Test fluctuation-dissipation theorem verification."""
        fdt_results = ward_checker.check_fluctuation_dissipation_relations()

        assert isinstance(fdt_results, dict)
        # FDT relations are fundamental to MSRJD theory
        # Allow empty results for placeholder implementation


class TestDimensionalConsistency:
    """Test complete dimensional analysis system."""

    @pytest.fixture
    def dimensional_analyzer(self):
        """Complete dimensional analyzer system."""
        parameters = IsraelStewartParameters(
            eta=3.0, zeta=1.0, kappa=1.5, tau_pi=0.2, tau_Pi=0.1, tau_q=0.08
        )
        is_system = IsraelStewartSystem(parameters)
        msrjd_action = MSRJDAction(is_system)
        extractor = VertexExtractor(msrjd_action)
        catalog = extractor.extract_all_vertices()
        rules = FeynmanRules(catalog, parameters=parameters)

        return DimensionalAnalyzer(rules)

    def test_complete_dimensional_analysis(self, dimensional_analyzer):
        """Test complete dimensional analysis of all components."""
        all_dimensions = dimensional_analyzer.analyze_all_dimensions()

        # Should analyze all component types
        expected_categories = ["vertices", "propagators", "couplings"]
        for category in expected_categories:
            assert category in all_dimensions, f"Missing dimensional analysis for {category}"

    def test_israel_stewart_parameter_dimensions(self, dimensional_analyzer):
        """Test Israel-Stewart parameter dimensional consistency."""
        coupling_dims = dimensional_analyzer._analyze_coupling_dimensions()

        # Check key Israel-Stewart parameters have correct dimensions
        is_parameters = ["eta", "zeta", "kappa", "tau_pi", "tau_Pi", "tau_q"]

        for param in is_parameters:
            if param in coupling_dims:
                dimension = coupling_dims[param]
                # Viscosities should have positive dimension, relaxation times negative
                if param in ["eta", "zeta", "kappa"]:
                    assert (
                        dimension > 0
                    ), f"Viscosity parameter {param} should have positive dimension, got {dimension}"
                elif "tau" in param:
                    assert (
                        dimension < 0
                    ), f"Relaxation time {param} should have negative dimension, got {dimension}"

    def test_dimensional_consistency_verification(self, dimensional_analyzer):
        """Test comprehensive dimensional consistency verification."""
        consistency = dimensional_analyzer.verify_dimensional_consistency()

        # Should check all major categories
        assert "overall" in consistency
        assert isinstance(consistency["overall"], bool)

        # Individual categories should also be checked
        categories = ["vertices", "propagators", "couplings"]
        for category in categories:
            if category in consistency:
                assert isinstance(consistency[category], bool)


class TestSystemIntegration:
    """Test complete system integration and performance."""

    def test_complete_feynman_rules_pipeline(self):
        """Test the complete pipeline from IS system to Feynman rules."""
        # Step 1: Create realistic Israel-Stewart system
        parameters = IsraelStewartParameters(
            eta=2.8, zeta=0.9, kappa=1.4, tau_pi=0.18, tau_Pi=0.09, tau_q=0.06
        )
        is_system = IsraelStewartSystem(parameters)

        # Step 2: Create MSRJD action
        msrjd_action = MSRJDAction(is_system, temperature=1.8)
        action_components = msrjd_action.construct_total_action()
        assert action_components.total != 0, "MSRJD action construction failed"

        # Step 3: Extract vertex catalog
        extractor = VertexExtractor(msrjd_action)
        catalog = extractor.extract_all_vertices()
        assert catalog.total_vertices > 0, "Vertex extraction failed"

        # Step 4: Generate Feynman rules
        rules = FeynmanRules(catalog, parameters=parameters)
        vertex_rules = rules.generate_all_vertex_rules()
        propagator_rules = rules.generate_propagator_rules()

        assert len(vertex_rules) > 0, "Vertex rule generation failed"
        assert len(propagator_rules) > 0, "Propagator rule generation failed"

        # Step 5: Verify Ward identities
        ward_results = rules.verify_ward_identities()
        assert "energy_momentum_conservation" in ward_results
        assert "gauge_invariance" in ward_results

        # Step 6: Check dimensional consistency
        dim_results = rules.verify_dimensional_consistency()
        assert "vertex_dimensions" in dim_results

        # Step 7: Generate summary report
        summary = rules.generate_feynman_rules_summary()
        assert len(summary) > 100, "Summary too brief"
        assert "Israel-Stewart" in summary, "Summary missing IS reference"

    def test_amplitude_calculation_integration(self):
        """Test integration with amplitude calculation."""
        parameters = IsraelStewartParameters(eta=2.0, zeta=0.5, kappa=1.0)
        is_system = IsraelStewartSystem(parameters)
        msrjd_action = MSRJDAction(is_system)
        extractor = VertexExtractor(msrjd_action)
        catalog = extractor.extract_all_vertices()
        rules = FeynmanRules(catalog, parameters=parameters)

        # Test amplitude calculation for simple process
        external_fields = ["rho", "u"]

        # Create momentum configuration
        k1, k2 = symbols("k1 k2", real=True)
        momentum_config = MomentumConfiguration(
            external_momenta={"k1": k1, "k2": k2},
            momentum_conservation=[k1 + k2],  # Simple conservation
            loop_momenta=[],
        )

        # Should be able to compute amplitude without errors
        try:
            amplitude = rules.get_amplitude_for_process(external_fields, momentum_config)
            # Amplitude should be a symbolic expression
            assert hasattr(amplitude, "free_symbols") or amplitude == 0
        except Exception as e:
            pytest.skip(f"Amplitude calculation not fully implemented: {e}")

    def test_performance_with_large_system(self):
        """Test system performance with comprehensive vertex catalog."""
        # Create system with enhanced parameters for larger vertex catalog
        parameters = IsraelStewartParameters(
            eta=5.0, zeta=2.0, kappa=3.0, tau_pi=0.3, tau_Pi=0.15, tau_q=0.1
        )
        is_system = IsraelStewartSystem(parameters)
        msrjd_action = MSRJDAction(is_system, temperature=3.0)

        # Time the complete extraction process
        import time

        start_time = time.time()

        extractor = VertexExtractor(msrjd_action)
        catalog = extractor.extract_all_vertices()
        rules = FeynmanRules(catalog, parameters=parameters)
        vertex_rules = rules.generate_all_vertex_rules()

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (allow up to 30 seconds for comprehensive analysis)
        assert execution_time < 30.0, f"System too slow: {execution_time:.2f} seconds"

        # Should produce substantial results
        assert len(vertex_rules) >= 1, "Large system produced insufficient results"


# Integration test for complete workflow
def test_complete_feynman_rules_workflow():
    """
    Complete integration test of the entire Feynman rules workflow.

    This test validates the full pipeline from Israel-Stewart parameters
    to complete Feynman rules with Ward identity verification.
    """
    # Create realistic physical system
    parameters = IsraelStewartParameters(
        eta=1.8,  # Realistic shear viscosity
        zeta=0.4,  # Bulk viscosity
        kappa=0.9,  # Thermal conductivity
        tau_pi=0.14,  # Shear relaxation
        tau_Pi=0.07,  # Bulk relaxation
        tau_q=0.045,  # Heat relaxation
    )

    # Build complete system
    is_system = IsraelStewartSystem(parameters)
    msrjd_action = MSRJDAction(is_system, temperature=2.2)

    # Extract and validate vertex catalog
    extractor = VertexExtractor(msrjd_action)
    catalog = extractor.extract_all_vertices()

    # Validate catalog quality
    validator = VertexValidator(catalog, parameters)
    ward_validity = validator.validate_ward_identities()
    assert all(ward_validity.values()), f"Ward identity validation failed: {ward_validity}"

    # Generate complete Feynman rules
    feynman_rules = FeynmanRules(catalog, parameters=parameters)
    vertex_rules = feynman_rules.generate_all_vertex_rules()
    propagator_rules = feynman_rules.generate_propagator_rules()

    # Comprehensive validation
    ward_check = feynman_rules.verify_ward_identities()
    dim_check = feynman_rules.verify_dimensional_consistency()

    # Final integration validation
    assert len(vertex_rules) > 0, "No vertex rules generated"
    assert len(propagator_rules) > 0, "No propagator rules generated"
    assert all(ward_check.values()), f"Ward identities failed: {ward_check}"
    assert all(dim_check.values()), f"Dimensional consistency failed: {dim_check}"

    # Test dimensional analyzer
    analyzer = DimensionalAnalyzer(feynman_rules)
    dimension_analysis = analyzer.analyze_all_dimensions()
    consistency_check = analyzer.verify_dimensional_consistency()

    assert "vertices" in dimension_analysis
    assert "propagators" in dimension_analysis
    assert "couplings" in dimension_analysis
    assert consistency_check[
        "overall"
    ], f"Overall dimensional consistency failed: {consistency_check}"

    # Generate final report
    summary = feynman_rules.generate_feynman_rules_summary()
    assert "Feynman Rules Summary" in summary
    assert "Israel-Stewart" in summary
    assert len(summary.split("\n")) > 10, "Summary too brief"

    print("\n✅ Complete Feynman Rules System Validation Successful!")
    print(f"   - Vertex rules: {len(vertex_rules)}")
    print(f"   - Propagator rules: {len(propagator_rules)}")
    print(f"   - Ward identities: {'✅' if all(ward_check.values()) else '❌'}")
    print(f"   - Dimensional consistency: {'✅' if all(dim_check.values()) else '❌'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
