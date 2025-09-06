"""
Pytest configuration and fixtures for the test suite
"""

from collections.abc import Generator

import numpy as np
import pytest

from rtrg.core import ISParameters, Metric
from rtrg.core.constants import PhysicalConstants
from rtrg.core.registry_factory import create_registry_for_context


@pytest.fixture
def metric() -> Metric:
    """Standard Minkowski metric fixture"""
    return Metric(dimension=4, signature=(-1, 1, 1, 1))


@pytest.fixture
def is_parameters() -> ISParameters:
    """Standard Israel-Stewart parameters for testing"""
    from rtrg.core.parameters import StandardParameterSets

    return StandardParameterSets.weakly_coupled_plasma(temperature=1.0)


@pytest.fixture
def field_registry(metric: Metric):
    """Field registry with all IS fields"""
    registry = create_registry_for_context("basic_physics", metric=metric)
    return registry


@pytest.fixture
def normalized_four_velocity() -> np.ndarray:
    """Normalized four-velocity in rest frame"""
    u = np.zeros(4)
    u[0] = PhysicalConstants.c  # u^0 = c in rest frame
    return u


@pytest.fixture
def spatial_velocity() -> np.ndarray:
    """Example spatial 3-velocity"""
    return np.array([0.1, 0.05, 0.0])  # Small velocity


@pytest.fixture
def sample_shear_tensor(metric: Metric) -> np.ndarray:
    """Sample symmetric traceless shear tensor"""
    pi = np.zeros((4, 4))

    # Make symmetric
    pi[1, 2] = pi[2, 1] = 0.1
    pi[1, 3] = pi[3, 1] = 0.05
    pi[2, 3] = pi[3, 2] = 0.02

    # Make traceless by subtracting trace part
    trace = np.trace(pi)
    pi -= trace / 4 * np.eye(4)

    return pi


@pytest.fixture
def numerical_tolerance() -> float:
    """Standard numerical tolerance for floating point comparisons"""
    return 1e-12


# Test parameter sets for different scenarios
@pytest.fixture(
    params=[
        {"eta": 0.1, "tau_pi": 0.01},
        {"eta": 0.5, "tau_pi": 0.001},
        {"eta": 0.01, "tau_pi": 0.1},
    ]
)
def parameter_variations(request) -> dict:
    """Various parameter combinations for testing robustness"""
    return request.param


# Marks for test categorization
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "physics: Physics validation tests")
    config.addinivalue_line("markers", "numerical: Numerical accuracy tests")
    config.addinivalue_line("markers", "slow: Slow tests")


# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle slow tests"""
    if config.getoption("--run-slow"):
        return  # Don't skip anything if explicitly requested

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
