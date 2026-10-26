from copy import deepcopy

import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.optics.refractive_index import build_refractive_index


@pytest.mark.parametrize("key,value,component", [
    ("n_550", 9., "real"), ("k_550", 2., "imag"),
    ("alpha_n", 1., "real"), ("alpha_k", 1., "imag"),
])
@pytest.mark.parametrize("copy_population", [False, True])
def test_attached_ri_callable_state_is_detached(key, value, component, copy_population):
    particle = make_particle(1e-7, ["BC"], [1.])
    wavelengths = np.array([400e-9, 550e-9, 800e-9])
    build_refractive_index(particle.species[0], wavelengths)
    original = ParticlePopulation(particle.species, np.array([particle.masses]), np.array([1.]), [0])
    clone = original.clone_detached() if copy_population else deepcopy(original)
    source_ri = original.species[0].refractive_index
    clone_ri = clone.species[0].refractive_index
    source_fun = getattr(source_ri, component + "_ri_fun")
    clone_fun = getattr(clone_ri, component + "_ri_fun")
    expected = clone_fun(wavelengths).copy()
    source_ri.RI_params[key] = value
    np.testing.assert_allclose(clone_fun(wavelengths), expected)
    assert not np.allclose(source_fun(wavelengths), expected)
    source_after = source_fun(wavelengths).copy()
    clone_ri.RI_params[key] = value * 2
    np.testing.assert_allclose(source_fun(wavelengths), source_after)
    assert not np.allclose(clone_fun(wavelengths), expected)


def test_parameter_overrides_retain_power_law_values():
    particle = make_particle(1e-7, ["BC"], [1.])
    wavelengths = np.array([400e-9, 550e-9, 800e-9])
    build_refractive_index(particle.species[0], wavelengths,
                           {"n_550": 1.9, "k_550": .8, "alpha_n": .2, "alpha_k": -.1})
    ri = particle.species[0].refractive_index
    np.testing.assert_allclose(ri.real_ri_fun(wavelengths), 1.9 * (wavelengths / 550e-9)**.2)
    np.testing.assert_allclose(ri.imag_ri_fun(wavelengths), .8 * (wavelengths / 550e-9)**-.1)
    np.testing.assert_array_equal(ri.real_ris, ri.real_ri_fun(wavelengths))
    np.testing.assert_array_equal(ri.imag_ris, ri.imag_ri_fun(wavelengths))
