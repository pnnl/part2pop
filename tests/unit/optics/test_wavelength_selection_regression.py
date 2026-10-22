from part2pop.optics.base import OpticalPopulation
from part2pop.population.base import ParticlePopulation
import numpy as np
import pytest


def test_select_505nm_does_not_match_500nm():
    base = ParticlePopulation(species=(), spec_masses=np.empty((1, 0)),
                              num_concs=np.array([1.]), ids=[1])
    pop = OpticalPopulation(base, [0.], [500e-9, 505e-9])
    pop.Cext[0, 0, :] = [7., 13.]
    assert pop.get_optical_coeff('b_ext', rh=0, wvl=505e-9) == 13.
    with pytest.raises(ValueError, match="Wavelength"):
        pop.get_optical_coeff('b_ext', rh=0, wvl=502e-9)


def test_off_grid_error_names_units_and_tolerance():
    base = ParticlePopulation(species=(), spec_masses=np.empty((1, 0)),
                              num_concs=np.array([1.]), ids=[1])
    pop = OpticalPopulation(base, [0.], [500e-9, 505e-9])
    # A request in nm (550) is a common mistake; the message should say why.
    with pytest.raises(ValueError, match=r"meters.*relative tolerance of 1e-07"):
        pop.get_optical_coeff('b_ext', rh=0, wvl=550)
