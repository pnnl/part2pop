"""Disabling activity must respect the retained solute rule and ice cutoff."""
import numpy as np
import pytest

from part2pop import Particle, get_species
from part2pop.freezing.builder import build_freezing_particle
from part2pop.freezing.factory.utils import calculate_Psat


@pytest.mark.parametrize("disabled", ["m_log10_Jhet", "b_log10_Jhet", "both"])
def test_disabled_oc_solute_rule_and_supersaturation_cutoff(disabled):
    specs = tuple(get_species(name, None) for name in ("OC", "BC", "H2O"))
    particle = Particle(specs, np.array([1e-21, 1e-21, 1e-20]) * [s.density for s in specs])
    coefficients = {"m_log10_Jhet": 0., "b_log10_Jhet": 0.}
    for name in coefficients:
        if disabled in (name, "both"):
            coefficients[name] = np.nan
    frozen = build_freezing_particle(particle, {"morphology": "homogeneous",
        "species_modifications": {"OC": coefficients,
                                  "BC": {"m_log10_Jhet": 0., "b_log10_Jhet": 12.}}})
    both = disabled == "both"
    expected_aw = (np.exp(-.01801528 * particle.masses[0]
                         / specs[0].molar_mass / particle.masses[2]) if both else 1.)
    assert frozen.get_aw() == pytest.approx(expected_aw, rel=1e-12, abs=0)
    vapor_pressure, ice_pressure = calculate_Psat(260.)
    assert (expected_aw < ice_pressure / vapor_pressure) == both
    assert frozen.compute_Jhet(260.) == pytest.approx(0. if both else 1e12, abs=0)
