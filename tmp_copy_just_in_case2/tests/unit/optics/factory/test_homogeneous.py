# tests/unit/optics/factory/test_core_shell.py

import numpy as np
import pytest

from part2pop.population.builder import build_population
from part2pop.optics.factory.homogeneous import HomogeneousParticle, build, _PMS_ERR


@pytest.mark.skipif(_PMS_ERR is not None, reason=f"PyMieScatt not available: {_PMS_ERR}")
def test_core_shell_particle_compute_optics():
    cfg_pop = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.2, 0.7, 0.1]],
    }
    pop = build_population(cfg_pop)
    base_particle = pop.get_particle(pop.ids[0])

    cfg = {"wvl_grid": [550e-9], "rh_grid": [0.0]}
    cp = HomogeneousParticle(base_particle, cfg)
    cp.compute_optics()

    assert np.isfinite(cp.Cext[0, 0])
    assert cp.Cext[0, 0] >= 0.0


@pytest.mark.skipif(_PMS_ERR is not None, reason=f"PyMieScatt not available: {_PMS_ERR}")
def test_core_shell_build_wrapper():
    cfg_pop = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.2, 0.7, 0.1]],
    }
    pop = build_population(cfg_pop)
    base_particle = pop.get_particle(pop.ids[0])
    cp = build(base_particle, {"wvl_grid": [550e-9], "rh_grid": [0.0]})
    assert isinstance(cp, HomogeneousParticle)


@pytest.mark.skipif(_PMS_ERR is not None, reason=f"PyMieScatt not available: {_PMS_ERR}")
def test_mixture_ri_handles_zero_volumes():
    cfg_pop = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.2, 0.7, 0.1]],
    }
    pop = build_population(cfg_pop)
    base_particle = pop.get_particle(pop.ids[0])
    hp = HomogeneousParticle(base_particle, {"wvl_grid": [550e-9], "rh_grid": [0.0]})
    hp.h2o_vols = np.zeros_like(hp.rh_grid)
    hp.dry_vol = 0.0
    assert hp._mixture_ri(0, 0) == complex(1.0, 0.0)
    arr = hp.get_cross_section("b_ext", rh_idx=0, wvl_idx=0)
    assert np.isfinite(arr)
    with pytest.raises(ValueError):
        hp.get_cross_section("unknown")
