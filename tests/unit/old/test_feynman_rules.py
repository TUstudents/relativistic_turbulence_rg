"""
Unit tests for the complete Feynman rules system.

Tests the integration of vertex extraction, Feynman rule generation,
Ward identity verification, and dimensional analysis for the relativistic
Israel-Stewart field theory.
"""

import pytest
import sympy as sp
from sympy import I, Symbol, symbols

from rtrg.core.constants import PhysicalConstants
from rtrg.core.tensors import Metric
from rtrg.field_theory.feynman_rules import (
    DimensionalAnalyzer,
    FeynmanRule,
    FeynmanRules,
    MomentumConfiguration,
    PropagatorRule,
    WardIdentityChecker,
)
from rtrg.field_theory.msrjd_action import MSRJDAction
from rtrg.field_theory.vertices import (
    VertexCatalog,
    VertexExtractor,
    VertexStructure,
)
from rtrg.israel_stewart.equations import IsraelStewartParameters, IsraelStewartSystem


@pytest.mark.unit
class TestMomentumConfiguration:
    """Test momentum configuration for Feynman rule evaluation."""

    def test_momentum_conservation_validation(self):
        """Test momentum conservation constraint validation."""
        k1, k2, k3 = symbols("k1 k2 k3")

        # Valid momentum conservation
        config = MomentumConfiguration(
            external_momenta={"k1": k1, "k2": k2, "k3": k3},
            momentum_conservation=[k1 + k2 - k3],  # k3 = k1 + k2
            loop_momenta=[],
        )

        # This should pass when k3 = k1 + k2
        # For now, just check structure is correct
        assert config.external_momenta["k1"] == k1
        assert len(config.momentum_conservation) == 1
        assert len(config.loop_momenta) == 0

    def test_loop_momentum_handling(self):
        """Test loop momentum specification."""
        k1, k2, q = symbols("k1 k2 q")

        config = MomentumConfiguration(
            external_momenta={"k1": k1, "k2": k2},
            momentum_conservation=[k1 + k2 - q],
            loop_momenta=[q],
        )

        assert "q" not in config.external_momenta
        assert q in config.loop_momenta


@pytest.mark.unit
class TestFeynmanRule:
    """Test individual Feynman rule functionality."""

    def test_feynman_rule_creation(self):
        """Test basic Feynman rule creation and validation."""
        omega, k = symbols("omega k", complex=True)
        eta = Symbol("eta", positive=True)

        rule = FeynmanRule(
            rule_type="vertex",
            fields=("u", "u_tilde", "pi"),
            amplitude=eta * k * omega,
            tensor_structure=sp.eye(3),
            momentum_factors=I * k,
            frequency_dependence=-I * omega,
            coupling_constants={eta},
            mass_dimension=4.0,
            symmetry_factor=1.0,
            hermiticity="none",
            ward_identities={"gauge_invariance": True},
            dimensional_consistency=True,
        )

        assert rule.rule_type == "vertex"
        assert len(rule.fields) == 3
        assert "u" in rule.fields
        assert eta in rule.coupling_constants
        assert rule.dimensional_consistency is True

    def test_momentum_evaluation(self):
        """Test evaluation at specific momentum configuration."""
        k1 = Symbol("k1")
        k = Symbol("k")

        rule = FeynmanRule(
            rule_type="vertex",
            fields=("u", "pi"),
            amplitude=k**2,  # Simple k-dependent amplitude
            tensor_structure=sp.eye(2),
            momentum_factors=k,
            frequency_dependence=sp.sympify(1),
            coupling_constants=set(),
            mass_dimension=2.0,
            symmetry_factor=1.0,
            hermiticity="hermitian",
            ward_identities={},
            dimensional_consistency=True,
        )

        momentum_config = MomentumConfiguration(
            external_momenta={"k": k1}, momentum_conservation=[], loop_momenta=[]
        )

        # Evaluate amplitude at specific momentum
        evaluated = rule.evaluate_at_momentum(momentum_config)
        expected = k1**2  # k^2 → k1^2

        # Check that substitution worked (simplified check)
        assert evaluated.subs(k1, 2) == 4


@pytest.mark.unit
class TestPropagatorRule:
    """Test propagator Feynman rule functionality."""

    def test_retarded_propagator_creation(self):
        """Test retarded propagator rule creation."""
        omega, k = symbols("omega k", complex=True)
        c_s = Symbol("c_s", positive=True)
        gamma = Symbol("gamma", positive=True)

        # Simple retarded propagator: G^R = 1/(ω - c_s k + iγ)
        dispersion = c_s * k
        pole = dispersion - I * gamma
        propagator_matrix = sp.Matrix([[1 / (omega - dispersion + I * gamma)]])

        rule = PropagatorRule(
            field_pair=("u", "u"),
            propagator_type="retarded",
            propagator_matrix=propagator_matrix,
            pole_structure=[pole],
            dispersion_relation=dispersion,
            damping_rates={"u": gamma},
        )

        assert rule.propagator_type == "retarded"
        assert rule.field_pair == ("u", "u")
        assert len(rule.pole_structure) == 1
        assert rule.verify_causality()  # Should pass for Im[pole] <= 0

    def test_spectral_function_extraction(self):
        """Test spectral function calculation."""
        omega, k = symbols("omega k", real=True)
        gamma = Symbol("gamma", positive=True, real=True)

        # Simple retarded propagator
        G_R = sp.Matrix([[1 / (omega + I * gamma)]])  # Pole at ω = -iγ

        rule = PropagatorRule(
            field_pair=("test", "test"),
            propagator_type="retarded",
            propagator_matrix=G_R,
            pole_structure=[],
            dispersion_relation=sp.sympify(0),
            damping_rates={},
        )

        # Extract spectral function
        spectral_func = rule.get_spectral_function()

        # Should be ρ(ω) = -2Im[G^R]/π = 2γ/(π(ω² + γ²))
        # Check that it's non-zero and has correct structure
        assert spectral_func != 0
        assert gamma in spectral_func.free_symbols

    def test_causality_verification(self):
        """Test causality check for propagator poles."""
        omega, k = symbols("omega k", complex=True)

        # Causal pole (Im[pole] ≤ 0)
        causal_pole = -I * Symbol("gamma", positive=True)
        causal_rule = PropagatorRule(
            field_pair=("u", "u"),
            propagator_type="retarded",
            propagator_matrix=sp.eye(1),
            pole_structure=[causal_pole],
            dispersion_relation=sp.sympify(0),
            damping_rates={},
        )

        assert causal_rule.verify_causality() is True

        # Non-causal pole (Im[pole] > 0)
        non_causal_pole = I * Symbol("gamma", positive=True)
        non_causal_rule = PropagatorRule(
            field_pair=("u", "u"),
            propagator_type="retarded",
            propagator_matrix=sp.eye(1),
            pole_structure=[non_causal_pole],
            dispersion_relation=sp.sympify(0),
            damping_rates={},
        )

        assert non_causal_rule.verify_causality() is False


@pytest.mark.unit
class TestFeynmanRulesIntegration:
    """Test integration of vertex extraction with Feynman rule generation."""

    def setup_method(self):
        """Set up test system with mock components."""
        # Create minimal IS parameters
        self.parameters = IsraelStewartParameters(
            eta=Symbol("eta", positive=True),
            tau_pi=Symbol("tau_pi", positive=True),
            zeta=Symbol("zeta", positive=True),
            tau_Pi=Symbol("tau_Pi", positive=True),
            kappa=Symbol("kappa", positive=True),
            tau_q=Symbol("tau_q", positive=True),
        )

        # Create mock vertex catalog
        eta, tau_pi = self.parameters.eta, self.parameters.tau_pi

        # Mock advection vertex: u_tilde * u * d_u
        advection_vertex = VertexStructure(
            fields=("u_tilde", "u", "u"),
            field_indices={"u_tilde": ["mu"], "u": ["nu", "alpha"]},
            coupling_expression=I * Symbol("k") * eta,
            tensor_structure="advection coupling",
            coupling_constants={eta},
            mass_dimension=4.0,
            derivative_structure={"u": 1},
            momentum_factors=I * Symbol("k"),
            symmetry_factor=1.0,
            vertex_type="advection",
        )

        # Mock shear vertex: pi_tilde * pi
        shear_vertex = VertexStructure(
            fields=("pi_tilde", "pi"),
            field_indices={"pi_tilde": ["mu", "nu"], "pi": ["mu", "nu"]},
            coupling_expression=1 / tau_pi,
            tensor_structure="relaxation coupling",
            coupling_constants={tau_pi},
            mass_dimension=4.0,
            derivative_structure={},
            momentum_factors=sp.sympify(1),
            symmetry_factor=1.0,
            vertex_type="stress",
        )

        self.vertex_catalog = VertexCatalog(
            three_point={("u_tilde", "u", "u"): advection_vertex},
            four_point={},
            constraint_vertices={("pi_tilde", "pi"): shear_vertex},
            total_vertices=2,
            coupling_constants={eta, tau_pi},
            vertex_types={"advection", "stress"},
        )

    def test_feynman_rules_generation(self):
        """Test complete Feynman rules generation from vertex catalog."""
        # Create Feynman rules system
        rules = FeynmanRules(vertex_catalog=self.vertex_catalog, parameters=self.parameters)

        # Generate vertex rules
        vertex_rules = rules.generate_all_vertex_rules()

        # Should have generated rules for our vertices
        assert len(vertex_rules) >= 1

        # Check specific rule properties
        for _field_combo, rule in vertex_rules.items():
            assert isinstance(rule, FeynmanRule)
            assert rule.rule_type == "vertex"
            assert len(rule.coupling_constants) > 0
            assert rule.dimensional_consistency is not None

    def test_propagator_rules_generation(self):
        """Test propagator rule generation."""
        rules = FeynmanRules(vertex_catalog=self.vertex_catalog, parameters=self.parameters)

        # Generate propagator rules
        propagator_rules = rules.generate_propagator_rules()

        # Should generate rules for physical fields
        assert len(propagator_rules) > 0

        # Check that retarded and Keldysh propagators are generated
        retarded_count = sum(1 for key in propagator_rules.keys() if "retarded" in str(key))
        keldysh_count = sum(1 for key in propagator_rules.keys() if "keldysh" in str(key))

        assert retarded_count > 0
        assert keldysh_count > 0

    def test_ward_identity_verification(self):
        """Test Ward identity verification system."""
        rules = FeynmanRules(vertex_catalog=self.vertex_catalog, parameters=self.parameters)

        # Generate rules first
        rules.generate_all_vertex_rules()
        rules.generate_propagator_rules()

        # Verify Ward identities
        ward_results = rules.verify_ward_identities()

        assert isinstance(ward_results, dict)
        assert "energy_momentum_conservation" in ward_results
        assert "current_conservation" in ward_results
        assert "gauge_invariance" in ward_results
        assert "causality" in ward_results

        # All checks should pass for our mock system
        assert all(ward_results.values())

    def test_dimensional_consistency_verification(self):
        """Test dimensional consistency verification."""
        rules = FeynmanRules(vertex_catalog=self.vertex_catalog, parameters=self.parameters)

        # Generate rules
        rules.generate_all_vertex_rules()

        # Check dimensional consistency
        dim_results = rules.verify_dimensional_consistency()

        assert isinstance(dim_results, dict)
        assert "vertex_dimensions" in dim_results
        assert "propagator_dimensions" in dim_results
        assert "coupling_dimensions" in dim_results

        # Should pass for our mock system
        assert dim_results["vertex_dimensions"] is True

    def test_feynman_rules_summary_generation(self):
        """Test comprehensive summary generation."""
        rules = FeynmanRules(vertex_catalog=self.vertex_catalog, parameters=self.parameters)

        # Generate all rules
        rules.generate_all_vertex_rules()
        rules.generate_propagator_rules()

        # Generate summary
        summary = rules.generate_feynman_rules_summary()

        assert isinstance(summary, str)
        assert "Feynman Rules Summary" in summary
        assert "Vertex Rules:" in summary
        assert "Propagator Rules:" in summary
        assert "Consistency Checks:" in summary

        # Should contain information about our mock vertices
        # The summary shows field combinations, not vertex types
        assert "u_tilde → u → u" in summary or "pi_tilde → pi" in summary


@pytest.mark.unit
class TestWardIdentityChecker:
    """Test Ward identity verification system."""

    def test_ward_identity_checker_initialization(self):
        """Test Ward identity checker setup."""
        # Create minimal system
        vertex_catalog = VertexCatalog(
            three_point={},
            four_point={},
            constraint_vertices={},
            total_vertices=0,
            coupling_constants=set(),
            vertex_types=set(),
        )

        rules = FeynmanRules(vertex_catalog)
        checker = WardIdentityChecker(rules)

        assert checker.rules == rules
        assert isinstance(checker.metric, Metric)

    def test_conservation_law_checks(self):
        """Test energy-momentum and current conservation checks."""
        vertex_catalog = VertexCatalog(
            three_point={},
            four_point={},
            constraint_vertices={},
            total_vertices=0,
            coupling_constants=set(),
            vertex_types=set(),
        )

        rules = FeynmanRules(vertex_catalog)
        checker = WardIdentityChecker(rules)

        # Test energy-momentum conservation
        em_results = checker.check_energy_momentum_conservation()
        assert isinstance(em_results, dict)

        # Test current conservation
        current_results = checker.check_current_conservation()
        assert isinstance(current_results, dict)

        # Test FDT relations
        fdt_results = checker.check_fluctuation_dissipation_relations()
        assert isinstance(fdt_results, dict)


@pytest.mark.unit
class TestDimensionalAnalyzer:
    """Test dimensional analysis system."""

    def test_dimensional_analyzer_initialization(self):
        """Test dimensional analyzer setup."""
        vertex_catalog = VertexCatalog(
            three_point={},
            four_point={},
            constraint_vertices={},
            total_vertices=0,
            coupling_constants=set(),
            vertex_types=set(),
        )

        rules = FeynmanRules(vertex_catalog)
        analyzer = DimensionalAnalyzer(rules)

        assert analyzer.rules == rules
        assert analyzer.natural_units is True

    def test_complete_dimensional_analysis(self):
        """Test complete dimensional analysis."""
        # Create system with some mock rules
        vertex_catalog = VertexCatalog(
            three_point={},
            four_point={},
            constraint_vertices={},
            total_vertices=0,
            coupling_constants=set(),
            vertex_types=set(),
        )

        parameters = IsraelStewartParameters(
            eta=Symbol("eta", positive=True),
            tau_pi=Symbol("tau_pi", positive=True),
            zeta=Symbol("zeta", positive=True),
            tau_Pi=Symbol("tau_Pi", positive=True),
            kappa=Symbol("kappa", positive=True),
            tau_q=Symbol("tau_q", positive=True),
        )

        rules = FeynmanRules(vertex_catalog, parameters=parameters)
        analyzer = DimensionalAnalyzer(rules)

        # Perform analysis
        results = analyzer.analyze_all_dimensions()

        assert isinstance(results, dict)
        assert "vertices" in results
        assert "propagators" in results
        assert "couplings" in results

        # Check coupling dimensions
        coupling_dims = results["couplings"]
        assert "eta" in coupling_dims
        assert "tau_pi" in coupling_dims
        assert coupling_dims["eta"] == -1.0  # Expected dimension in natural units


@pytest.mark.integration
class TestFeynmanRulesSystemIntegration:
    """Integration test for complete Feynman rules system."""

    def test_full_system_workflow(self):
        """Test complete workflow from action to Feynman rules."""
        # This would test the full pipeline:
        # 1. Create IS system and MSRJD action
        # 2. Extract vertices using VertexExtractor
        # 3. Generate Feynman rules
        # 4. Verify Ward identities and dimensions
        # 5. Generate summary

        # For now, test that all components can work together
        parameters = IsraelStewartParameters(
            eta=Symbol("eta", positive=True),
            tau_pi=Symbol("tau_pi", positive=True),
            zeta=Symbol("zeta", positive=True),
            tau_Pi=Symbol("tau_Pi", positive=True),
            kappa=Symbol("kappa", positive=True),
            tau_q=Symbol("tau_q", positive=True),
        )

        # Create minimal vertex catalog
        vertex_catalog = VertexCatalog(
            three_point={},
            four_point={},
            constraint_vertices={},
            total_vertices=0,
            coupling_constants=set(),
            vertex_types=set(),
        )

        # Create Feynman rules system
        rules = FeynmanRules(vertex_catalog, parameters=parameters)

        # Generate all components
        vertex_rules = rules.generate_all_vertex_rules()
        propagator_rules = rules.generate_propagator_rules()
        ward_check = rules.verify_ward_identities()
        dim_check = rules.verify_dimensional_consistency()
        summary = rules.generate_feynman_rules_summary()

        # Verify all components exist and are consistent
        assert isinstance(vertex_rules, dict)
        assert isinstance(propagator_rules, dict)
        assert isinstance(ward_check, dict)
        assert isinstance(dim_check, dict)
        assert isinstance(summary, str)

        # Summary should contain basic information
        assert "Feynman Rules Summary" in summary

    def test_amplitude_calculation_framework(self):
        """Test framework for amplitude calculation."""
        vertex_catalog = VertexCatalog(
            three_point={},
            four_point={},
            constraint_vertices={},
            total_vertices=0,
            coupling_constants=set(),
            vertex_types=set(),
        )

        rules = FeynmanRules(vertex_catalog)

        # Create momentum configuration
        k1, k2 = symbols("k1 k2")
        momentum_config = MomentumConfiguration(
            external_momenta={"k1": k1, "k2": k2}, momentum_conservation=[k1 + k2], loop_momenta=[]
        )

        # Try to compute amplitude (should return zero for empty catalog)
        amplitude = rules.get_amplitude_for_process(
            external_fields=["u", "pi"], momentum_config=momentum_config
        )

        assert amplitude == 0  # Empty catalog should give zero amplitude
