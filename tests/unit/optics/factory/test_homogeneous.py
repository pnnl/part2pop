# tests/unit/optics/factory/test_homogeneous.py

import numpy as np

from pyparticle.population.builder import build_population
from pyparticle.optics.factory.homogeneous import HomogeneousParticle, build


def _make_monodisperse_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.9, 0.1]],
    }
    return build_population(cfg)


def test_homogeneous_particle_compute_optics():
    pop = _make_monodisperse_population()
    base_particle = pop.get_particle(pop.ids[0])

    cfg = {"wvl_grid": [550e-9], "rh_grid": [0.0]}
    hp = HomogeneousParticle(base_particle, cfg)
    hp.compute_optics()

    assert np.isfinite(hp.Cext[0, 0])
    assert hp.Cext[0, 0] >= 0.0


def test_homogeneous_build_wrapper():
    pop = _make_monodisperse_population()
    base_particle = pop.get_particle(pop.ids[0])
    hp = build(base_particle, {"wvl_grid": [550e-9], "rh_grid": [0.0]})
    assert isinstance(hp, HomogeneousParticle)
