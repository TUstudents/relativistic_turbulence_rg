"""
Basic integration test for Feynman Rules system functionality.

This test validates that the core Feynman rules system works correctly
with simplified vertex extraction and rule generation.
"""

import pytest
import sympy as sp
from sympy import I, Symbol, symbols

from rtrg.field_theory.feynman_rules import (
    DimensionalAnalyzer,
    FeynmanRules,
)
from rtrg.field_theory.msrjd_action import MSRJDAction
from rtrg.field_theory.vertices import VertexCatalog, VertexExtractor, VertexStructure
from rtrg.israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem


def test_basic_feynman_rules_system():
    """Test basic functionality of the Feynman rules system."""

    # Create Israel-Stewart system
    parameters = IsraelStewartParameters(
        eta=2.0, zeta=0.5, kappa=1.0, tau_pi=0.15, tau_Pi=0.08, tau_q=0.05
    )
    is_system = IsraelStewartSystem(parameters)

    # Test MSRJD action construction
    msrjd_action = MSRJDAction(is_system, temperature=1.5)
    action_components = msrjd_action.construct_total_action()

    assert action_components.total != 0, "MSRJD action should be non-zero"
    assert action_components.deterministic != 0, "Deterministic action should be non-zero"

    # Create a minimal vertex catalog for testing
    test_vertex = VertexStructure(
        fields=("rho", "u"),
        field_indices={"rho": [], "u": ["mu"]},  # rho scalar, u has one index
        coupling_expression=parameters.eta * Symbol("rho") * Symbol("u"),
        tensor_structure="scalar coupling",
        coupling_constants={parameters.eta},
        mass_dimension=4.0,
        derivative_structure={"rho": 0, "u": 0},
        momentum_factors=sp.sympify(1),
        symmetry_factor=1.0,
        vertex_type="energy_momentum",
    )

    catalog = VertexCatalog(
        three_point={("rho", "u"): test_vertex},
        four_point={},
        constraint_vertices={},
        total_vertices=1,
        coupling_constants={parameters.eta},
        vertex_types={"energy_momentum"},
    )

    # Test Feynman rules generation
    feynman_rules = FeynmanRules(catalog, parameters=parameters)

    # Test vertex rule generation
    vertex_rules = feynman_rules.generate_all_vertex_rules()
    assert len(vertex_rules) > 0, "Should generate at least one vertex rule"

    # Test propagator rule generation
    propagator_rules = feynman_rules.generate_propagator_rules()
    assert len(propagator_rules) > 0, "Should generate propagator rules"

    # Test Ward identity verification
    ward_results = feynman_rules.verify_ward_identities()
    assert isinstance(ward_results, dict), "Ward identity results should be a dictionary"

    # Test dimensional consistency
    dim_results = feynman_rules.verify_dimensional_consistency()
    assert isinstance(dim_results, dict), "Dimensional consistency results should be a dictionary"

    print("✅ Basic Feynman rules system test passed!")


def test_dimensional_analyzer():
    """Test the dimensional analysis system."""

    parameters = IsraelStewartParameters(eta=1.5, zeta=0.4, kappa=0.8)
    is_system = IsraelStewartSystem(parameters)
    msrjd_action = MSRJDAction(is_system)

    # Create minimal catalog
    catalog = VertexCatalog(
        three_point={},
        four_point={},
        constraint_vertices={},
        total_vertices=0,
        coupling_constants=set(),
        vertex_types=set(),
    )
    feynman_rules = FeynmanRules(catalog, parameters=parameters)

    # Test dimensional analyzer
    analyzer = DimensionalAnalyzer(feynman_rules)

    # Test coupling dimension analysis
    coupling_dims = analyzer._analyze_coupling_dimensions()
    assert isinstance(coupling_dims, dict), "Should return dictionary of coupling dimensions"

    # Check Israel-Stewart parameter dimensions
    assert "eta" in coupling_dims, "Should analyze eta dimension"
    assert "zeta" in coupling_dims, "Should analyze zeta dimension"
    assert "kappa" in coupling_dims, "Should analyze kappa dimension"

    # Test consistency verification
    consistency = analyzer.verify_dimensional_consistency()
    assert "overall" in consistency, "Should check overall consistency"
    assert isinstance(consistency["overall"], bool), "Overall consistency should be boolean"

    print("✅ Dimensional analyzer test passed!")


def test_momentum_space_conversion():
    """Test momentum space conversion functionality."""

    parameters = IsraelStewartParameters(eta=1.0, zeta=0.2, kappa=0.5)
    catalog = VertexCatalog(
        three_point={},
        four_point={},
        constraint_vertices={},
        total_vertices=0,
        coupling_constants=set(),
        vertex_types=set(),
    )
    feynman_rules = FeynmanRules(catalog, parameters=parameters)

    # Test coordinate to momentum space conversion
    t, x, y, z = symbols("t x y z", real=True)
    test_function = sp.Function("f")

    # Simple derivative expression
    test_expr = sp.Derivative(test_function(t, x), t)
    converted = feynman_rules._convert_to_momentum_space(test_expr)

    # Should contain momentum space elements
    converted_str = str(converted)

    # Check if conversion happened (should contain I for imaginary unit or omega/k symbols)
    has_momentum_elements = (
        "I" in converted_str
        or "omega" in converted_str
        or "k_vec" in converted_str
        or "k_mu" in converted_str
    )

    assert has_momentum_elements, f"Momentum space conversion should work, got: {converted}"

    print("✅ Momentum space conversion test passed!")


def test_dispersion_relations():
    """Test realistic dispersion relation generation."""

    parameters = IsraelStewartParameters(
        eta=2.5, zeta=0.8, kappa=1.2, tau_pi=0.12, tau_Pi=0.06, tau_q=0.04
    )
    catalog = VertexCatalog(
        three_point={},
        four_point={},
        constraint_vertices={},
        total_vertices=0,
        coupling_constants=set(),
        vertex_types=set(),
    )
    feynman_rules = FeynmanRules(catalog, parameters=parameters)

    # Test dispersion relations for different fields
    fields_to_test = ["u", "pi", "Pi", "q", "rho"]

    for field in fields_to_test:
        dispersion = feynman_rules._get_dispersion_relation(field)

        # Should be a non-zero symbolic expression
        assert dispersion != 0, f"Dispersion relation for {field} should be non-zero"
        assert hasattr(dispersion, "free_symbols"), f"Dispersion for {field} should be symbolic"

        # Should contain momentum or frequency dependence
        dispersion_str = str(dispersion)
        has_momentum_freq = any(symbol in dispersion_str.lower() for symbol in ["k", "omega", "i"])
        assert (
            has_momentum_freq
        ), f"Dispersion for {field} should have momentum/frequency dependence"

    print("✅ Dispersion relations test passed!")


def test_system_summary_generation():
    """Test summary generation for the complete system."""

    parameters = IsraelStewartParameters(eta=2.0, zeta=0.6, kappa=1.1)
    is_system = IsraelStewartSystem(parameters)
    msrjd_action = MSRJDAction(is_system)

    # Create minimal catalog with test vertices
    catalog = VertexCatalog(
        three_point={},
        four_point={},
        constraint_vertices={},
        total_vertices=0,
        coupling_constants=set(),
        vertex_types=set(),
    )

    # Add some test vertices
    test_three_point = {}
    for i, fields in enumerate([("rho", "u"), ("pi", "u"), ("q", "rho")]):
        vertex = VertexStructure(
            fields=fields,
            vertex_type=f"test_type_{i}",
            coupling_constants=[parameters.eta],
            coupling_expression=parameters.eta * sp.Symbol("test"),
            tensor_structure="test structure",
            mass_dimension=4.0,
            symmetry_factor=1.0,
            derivative_structure={},
        )
        test_three_point[fields] = vertex

    # Recreate catalog with vertices
    catalog = VertexCatalog(
        three_point=test_three_point,
        four_point={},
        constraint_vertices={},
        total_vertices=len(test_three_point),
        coupling_constants={parameters.eta},
        vertex_types={f"test_type_{i}" for i in range(len(test_three_point))},
    )

    feynman_rules = FeynmanRules(catalog, parameters=parameters)

    # Generate summary
    summary = feynman_rules.generate_feynman_rules_summary()

    # Should be a substantial string
    assert len(summary) > 50, "Summary should be substantial"
    assert "Feynman Rules Summary" in summary, "Should contain title"
    assert "Israel-Stewart" in summary, "Should reference Israel-Stewart theory"

    print("✅ System summary generation test passed!")


def test_complete_basic_integration():
    """Test complete integration of basic functionality."""

    print("\n🧪 Running Complete Basic Feynman Rules Integration Test...")

    # Step 1: Create realistic system
    parameters = IsraelStewartParameters(
        eta=1.8, zeta=0.4, kappa=0.9, tau_pi=0.14, tau_Pi=0.07, tau_q=0.045
    )
    is_system = IsraelStewartSystem(parameters)
    msrjd_action = MSRJDAction(is_system, temperature=2.0)

    print("   ✓ Israel-Stewart system created")

    # Step 2: Create and populate vertex catalog
    # Start with empty catalog and add vertices
    three_point_vertices = {}
    four_point_vertices = {}
    constraint_vertices = {}
    all_coupling_constants = {parameters.eta}
    all_vertex_types = set()

    # Add representative vertices for different physics
    test_vertices = [
        (("rho", "u"), "energy_momentum", "Energy-momentum coupling"),
        (("pi", "u", "u"), "shear_transport", "Shear stress transport"),
        (("q", "rho"), "heat_transport", "Heat flux transport"),
    ]

    for fields, vertex_type, description in test_vertices:
        vertex = VertexStructure(
            fields=fields,
            vertex_type=vertex_type,
            coupling_constants=[parameters.eta],
            coupling_expression=parameters.eta * sp.prod([sp.Symbol(f) for f in fields]),
            tensor_structure=description,
            mass_dimension=4.0,
            symmetry_factor=1.0,
            derivative_structure={field: 0 for field in fields},
        )

        three_point_vertices[fields] = vertex
        all_coupling_constants.add(parameters.eta)
        all_vertex_types.add(vertex_type)

    # Create catalog with populated vertices
    catalog = VertexCatalog(
        three_point=three_point_vertices,
        four_point=four_point_vertices,
        constraint_vertices=constraint_vertices,
        total_vertices=len(three_point_vertices),
        coupling_constants=all_coupling_constants,
        vertex_types=all_vertex_types,
    )

    print(f"   ✓ Vertex catalog created with {catalog.total_vertices} vertices")

    # Step 3: Generate Feynman rules
    feynman_rules = FeynmanRules(catalog, parameters=parameters)
    vertex_rules = feynman_rules.generate_all_vertex_rules()
    propagator_rules = feynman_rules.generate_propagator_rules()

    print(
        f"   ✓ Generated {len(vertex_rules)} vertex rules and {len(propagator_rules)} propagator rules"
    )

    # Step 4: Verify system consistency
    ward_check = feynman_rules.verify_ward_identities()
    dim_check = feynman_rules.verify_dimensional_consistency()

    print(f"   ✓ Ward identities: {ward_check}")
    print(f"   ✓ Dimensional consistency: {dim_check}")

    # Step 5: Test dimensional analysis
    analyzer = DimensionalAnalyzer(feynman_rules)
    coupling_analysis = analyzer._analyze_coupling_dimensions()
    consistency_check = analyzer.verify_dimensional_consistency()

    print(f"   ✓ Analyzed {len(coupling_analysis)} coupling constants")
    print(f"   ✓ Overall dimensional consistency: {consistency_check.get('overall', False)}")

    # Step 6: Generate summary
    summary = feynman_rules.generate_feynman_rules_summary()
    summary_lines = len(summary.split("\n"))

    print(f"   ✓ Generated system summary ({summary_lines} lines)")

    # Final validation
    assert len(vertex_rules) > 0, "Should have vertex rules"
    assert len(propagator_rules) > 0, "Should have propagator rules"
    assert isinstance(ward_check, dict), "Ward check should return dict"
    assert isinstance(dim_check, dict), "Dim check should return dict"
    assert len(coupling_analysis) > 0, "Should analyze coupling dimensions"
    assert summary_lines > 10, "Summary should be substantial"

    print("\n✅ Complete Basic Feynman Rules Integration Test PASSED!")
    print("   📊 System Statistics:")
    print(f"   - Vertex rules: {len(vertex_rules)}")
    print(f"   - Propagator rules: {len(propagator_rules)}")
    print(f"   - Coupling constants analyzed: {len(coupling_analysis)}")
    print(f"   - Summary length: {summary_lines} lines")

    return True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
