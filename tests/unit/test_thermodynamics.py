"""
Unit tests for thermodynamics and EOS implementations.
"""

import numpy as np
import pytest

from rtrg.israel_stewart.thermodynamics import ConformalEOS, IdealGasEOS


@pytest.mark.unit
class TestConformalEOS:
    def test_pressure_and_cs2(self):
        eos = ConformalEOS()
        rho = 3.0
        assert eos.pressure(rho) == pytest.approx(1.0)
        assert eos.cs2(rho) == pytest.approx(1.0 / 3.0)

    def test_temperature_scaling(self):
        eos = ConformalEOS()
        rhos = [0.1, 1.0, 10.0]
        temps = [eos.temperature(r) for r in rhos]
        assert temps[2] > temps[1] > temps[0]


@pytest.mark.unit
class TestIdealGasEOS:
    def test_gamma_bounds(self):
        with pytest.raises(ValueError):
            IdealGasEOS(gamma=0.9)
        with pytest.raises(ValueError):
            IdealGasEOS(gamma=2.5)

    def test_pressure_cs2(self):
        eos = IdealGasEOS(gamma=4.0 / 3.0)
        rho = 3.0
        assert eos.pressure(rho) == pytest.approx((4.0 / 3.0 - 1.0) * rho)
        assert eos.cs2(rho) == pytest.approx(1.0 / 3.0)

    def test_temperature_reference_scaling(self):
        eos = IdealGasEOS(gamma=4.0 / 3.0, T0=2.0, rho0=2.0)
        # Keep proportional to reference at rho = rho0
        assert eos.temperature(eos.rho0) == pytest.approx(eos.T0)
        # Monotonic with rho (for this simple model)
        assert eos.temperature(4.0) > eos.temperature(2.0)
