import numpy as np
import pytest
from part2pop.population.factory import sampled_lognormals


def _cfg(n_parts, concentrations=(1., 1., 1.)):
    return {
        "N": list(concentrations), "GMD": [1e-7] * len(concentrations),
        "GSD": [1.2] * len(concentrations), "N_parts": n_parts,
        "aero_spec_names": [["SO4"]] * len(concentrations),
        "aero_spec_fracs": [[1.0]] * len(concentrations), "seed": 7,
    }


def test_fewer_samples_than_modes_raises_instead_of_hanging():
    with pytest.raises(ValueError, match="at least the number of modes"):
        sampled_lognormals.build(_cfg(2))


@pytest.mark.parametrize("count", [3, 4, 10, 11])
def test_allocation_has_exact_total_and_keeps_every_mode(count):
    pop = sampled_lognormals.build(_cfg(count))
    assert len(pop.ids) == count
    assert np.isclose(pop.get_Ntot(), 3.)


def test_invalid_mode_concentrations_raise():
    for conc in [(0., 0., 0.), (-1., 2., 1.), (float('nan'), 1., 1.)]:
        with pytest.raises(ValueError, match="Mode number concentrations"):
            sampled_lognormals.build(_cfg(6, conc))


def test_finite_concentrations_with_overflowing_total_are_rejected():
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="finite positive total"):
            sampled_lognormals.build(_cfg(10, (1e308, 1e308, 1e308)))


def test_large_finite_total_keeps_exact_requested_count():
    pop = sampled_lognormals.build(_cfg(10, (1e307, 1e307, 1e307)))
    assert len(pop.ids) == 10
    assert pop.get_Ntot() == pytest.approx(3e307)
