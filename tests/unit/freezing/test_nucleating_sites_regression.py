import numpy as np
import pytest
from part2pop.freezing.base import FreezingPopulation
from part2pop.population.base import ParticlePopulation


def _population():
    base = ParticlePopulation(species=(), spec_masses=np.empty((2, 0)),
                              num_concs=np.array([1., 3.]), ids=[1, 2])
    pop = FreezingPopulation(base)
    pop.INSA[:] = [1., 3.]
    pop.compute_Jhets = lambda T, cfg: np.array([2., 4.])
    return pop


def test_surface_area_weighted_cooling_integral():
    pop = _population()
    # Mean Jhet: (1*1*2 + 3*3*4)/(1*1 + 3*3) = 3.8 m^-2 s^-1.
    assert pop.get_nucleating_sites([260., 250., 240.], 2., {}) == pytest.approx([0., 19., 38.])


def test_celsius_and_kelvin_are_equivalent():
    pop = _population()
    c = pop.get_nucleating_sites([-13.15, -23.15], 2., {}, T_units="C")
    k = pop.get_nucleating_sites([260., 250.], 2., {}, T_units="K")
    np.testing.assert_allclose(c, k)


@pytest.mark.parametrize("grid,rate", [([250., 260.], 1.), ([260., 250.], 0.), ([260., 260.], 1.)])
def test_rejects_invalid_cooling_conditions(grid, rate):
    with pytest.raises(ValueError):
        _population().get_nucleating_sites(grid, rate, {})
