import numpy as np
import pytest
from part2pop.freezing.builder import build_freezing_particle

from part2pop import Particle, get_species


def freezing(names, masses):
    p = Particle(tuple(get_species(n, None) for n in names), masses)
    return build_freezing_particle(p, {"morphology": "homogeneous"})


def test_equivalent_salt_and_ions_have_analytic_ideal_activity():
    salt = freezing(["NaCl", "H2O"], [5.85e-19, 1e-17])
    ions = freezing(["Na", "Cl", "H2O"], [2.3e-19, 3.55e-19, 1e-17])
    expected = np.exp(-0.01801528 * 2.0)
    assert salt.get_aw() == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert ions.get_aw() == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert salt.compute_Jhom(235.0) == pytest.approx(
        ions.compute_Jhom(235.0), rel=1e-12, abs=0.0
    )


def test_order_and_already_separated_ions_are_not_double_counted():
    p = freezing(["H2O", "Cl", "Na"], [1e-17, 3.55e-19, 2.3e-19])
    assert p.get_aw() == pytest.approx(np.exp(-0.01801528 * 2.0), rel=1e-12, abs=0.0)


def test_pure_water_other_solute_and_dry_rate_controls():
    assert freezing(["H2O"], [1e-17]).get_aw() == 1.0
    sulfate = freezing(["SO4", "H2O"], [0.096e-17, 1e-17])
    assert sulfate.get_aw() == pytest.approx(np.exp(-0.01801528), rel=1e-12, abs=0.0)
    dry = freezing(["NaCl", "H2O"], [5.85e-19, 0.0])
    assert dry.compute_Jhom(235.0) == 0.0


@pytest.mark.parametrize("molality", [0.01, 1.0, 5.0])
@pytest.mark.parametrize("scale", [1.0, 1e9])
def test_ideal_convention_is_invariant_across_dilution_and_particle_mass_scale(
    molality, scale
):
    water = 1e-17 * scale
    salt = freezing(["NaCl", "H2O"], [0.0585 * molality * water, water])
    ions = freezing(
        ["Na", "Cl", "H2O"],
        [0.023 * molality * water, 0.0355 * molality * water, water],
    )
    expected = np.exp(-0.01801528 * 2 * molality)
    assert salt.get_aw() == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert ions.get_aw() == pytest.approx(expected, rel=1e-12, abs=0.0)
    for temperature in (235.0, 250.0):
        assert salt.compute_Jhom(temperature) == pytest.approx(
            ions.compute_Jhom(temperature), rel=1e-12, abs=0.0
        )


def test_mixed_salt_ions_and_other_solute_are_counted_once_each():
    water = 1e-17
    # 0.5 molal NaCl + 0.5 molal Na and Cl + 1 molal sulfate = 3 osmoles/kg.
    mixed = freezing(
        ["nacl", "Na", "Cl", "SO4", "H2O"],
        [
            0.0585 * 0.5 * water,
            0.023 * 0.5 * water,
            0.0355 * 0.5 * water,
            0.096 * water,
            water,
        ],
    )
    assert mixed.get_aw() == pytest.approx(
        np.exp(-0.01801528 * 3.0), rel=1e-12, abs=0.0
    )
