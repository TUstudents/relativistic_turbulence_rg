"""
Benchmarks for dispersion relations in linearized Israel–Stewart hydrodynamics.
"""

import numpy as np
import pytest

from rtrg.israel_stewart.equations import IsraelStewartParameters
from rtrg.israel_stewart.linearized import BackgroundState, LinearizedIS
from rtrg.israel_stewart.thermodynamics import ConformalEOS


@pytest.mark.benchmark
@pytest.mark.physics
def test_sound_wave_dispersion_low_k():
    # Parameters (moderate viscosities)
    params = IsraelStewartParameters(
        eta=0.1,
        zeta=0.05,
        kappa=0.2,
        tau_pi=0.1,
        tau_Pi=0.05,
        tau_q=0.05,
        temperature=1.0,
    )
    background = BackgroundState(rho=1.0, pressure=1.0 / 3.0)
    eos = ConformalEOS()
    system = LinearizedIS(background, params, eos=eos)

    # k grid near zero
    k_vals = np.linspace(0.01, 0.2, 10)
    cs = np.sqrt(eos.cs2(background.rho))

    # Check Re(ω) ≈ c_s k
    for k in k_vals:
        omega = system.dispersion_relation(k, mode="sound")
        assert abs(omega.real - cs * k) < 0.5 * cs * k + 1e-6


@pytest.mark.benchmark
@pytest.mark.physics
def test_sound_attenuation_coefficient():
    # Parameters and background
    params = IsraelStewartParameters(
        eta=0.1,
        zeta=0.05,
        kappa=0.2,
        tau_pi=0.1,
        tau_Pi=0.05,
        tau_q=0.05,
        temperature=1.0,
    )
    background = BackgroundState(rho=2.0, pressure=2.0 / 3.0)
    system = LinearizedIS(background, params, eos=ConformalEOS())

    gamma = system.sound_attenuation_coefficient()
    expected = (4 * params.eta / 3 + params.zeta) / background.rho
    assert abs(gamma - expected) < 1e-12
