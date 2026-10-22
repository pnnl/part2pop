"""Reject nonfinite active-material coefficients without changing valid rates."""
import numpy as np
import pytest

from part2pop import make_particle
from part2pop.freezing.builder import build_freezing_particle


def test_nonfinite_coefficients_do_not_enter_rate_average():
    particle = make_particle(100e-9, ["BC", "H2O"], [0.1, 0.9])
    frozen = build_freezing_particle(particle, {"morphology": "homogeneous"})
    assert frozen.compute_Jhet(250.0) > 0
    for bad in [np.inf, -np.inf, np.nan]:
        for field in ("m_log10_Jhet", "b_log10_Jhet"):
            original = getattr(frozen, field)[0]
            getattr(frozen, field)[0] = bad
            try:
                assert frozen.compute_Jhet(250.0) == 0.0
            finally:
                getattr(frozen, field)[0] = original
