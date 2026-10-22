"""Nonfinite heterogeneous coefficients cannot supply ice-active area."""
import numpy as np
import pytest

from part2pop import Particle, get_species
from part2pop.freezing.builder import build_freezing_particle


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["m_log10_Jhet", "b_log10_Jhet"])
def test_nonfinite_coefficient_excludes_active_area(bad, field):
    particle = Particle((get_species("BC", None), get_species("H2O", None)),
                        [1e-18, 1e-17])
    modifications = {"BC": {field: bad}}
    frozen = build_freezing_particle(particle, {
        "morphology": "homogeneous", "species_modifications": modifications
    })
    assert frozen.INSA == 0.0
