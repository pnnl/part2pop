# tests/unit/optics/test_builder.py

import numpy as np
import pytest

from pyparticle.population.builder import build_population
from pyparticle.population.base import ParticlePopulation
from pyparticle.optics.builder import (
    build_optical_particle,
    build_optical_population,
)


def _make_monodisperse_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "BC", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.7, 0.2, 0.1]],
    }
    pop = build_population(cfg)
    assert isinstance(pop, ParticlePopulation)
    return pop


def test_build_optical_particle_missing_type_raises():
    pop = _make_monodisperse_population()
    base_particle = pop.get_particle(pop.ids[0])

    with pytest.raises(ValueError):
        build_optical_particle(base_particle, {})


def test_build_optical_population_homogeneous():
    pop = _make_monodisperse_population()
    cfg = {
        "type": "homogeneous",
        "wvl_grid": [550e-9],
        "rh_grid": [0.0],
    }

    optical_pop = build_optical_population(pop, cfg)

    assert isinstance(optical_pop.base_population, ParticlePopulation)
    assert hasattr(optical_pop, "get_optical_coeff")

    b_ext = optical_pop.get_optical_coeff("b_ext", rh=0, wvl=550e-9)
    assert np.isfinite(b_ext)
    assert b_ext >= 0.0
