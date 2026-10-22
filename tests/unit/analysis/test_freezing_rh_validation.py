import copy

import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.builder import build_variable


def population():
    p = make_particle(1e-7, ["BC", "SO4"], [.5, .5])
    return ParticlePopulation(p.species, np.array([p.masses]), np.array([1.]), [0],
                              species_modifications={"BC": {"b_log10_Jhet": 4.}})


def run(name, pop, rh):
    cfg = {"RH": rh, "species_modifications": {"BC": {"b_log10_Jhet": 8.}}}
    if name == "P_frz":
        return build_variable(name, scope="particle", var_cfg=dict(cfg, T=250., T_units="K")).compute_all(pop)
    return build_variable(name, var_cfg=dict(cfg, T_grid=[250.], T_units="K")).compute(pop)


@pytest.mark.parametrize("name", ["P_frz", "avg_Jhet"])
@pytest.mark.parametrize("rh", [85., -0.1, 1.1, np.nan, np.inf, -np.inf, None, [0.85]])
def test_invalid_rh_rejected_before_population_changes(name, rh):
    pop = population()
    masses = pop.spec_masses.copy()
    mods = copy.deepcopy(pop.species_modifications)
    with pytest.raises(ValueError, match="RH must be"):
        run(name, pop, rh)
    np.testing.assert_array_equal(pop.spec_masses, masses)
    assert pop.species_modifications == mods


@pytest.mark.parametrize("name", ["P_frz", "avg_Jhet"])
def test_saturation_input_retains_documented_099_equilibrium(name):
    first, second = population(), population()
    result099, result1 = run(name, first, .99), run(name, second, 1.)
    np.testing.assert_array_equal(first.spec_masses, second.spec_masses)
    np.testing.assert_array_equal(result099, result1)


@pytest.mark.parametrize("name", ["P_frz", "avg_Jhet"])
def test_zero_rh_remains_valid_and_dries_water(name):
    pop = population()
    run(name, pop, 0.)
    assert pop.get_particle(0).get_mass_h2o() == 0
