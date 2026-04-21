# tests/unit/optics/factory/test_core_shell.py

import numpy as np
import pytest

from part2pop.population.builder import build_population
from part2pop.optics.factory.core_shell import CoreShellParticle, build, _PMS_ERR


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
    cp = CoreShellParticle(base_particle, cfg)
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
    assert isinstance(cp, CoreShellParticle)


@pytest.mark.skipif(_PMS_ERR is not None, reason=f"PyMieScatt not available: {_PMS_ERR}")
def test_shell_ri_returns_default_when_volumes_zero():
    cfg_pop = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.2, 0.7, 0.1]],
    }
    pop = build_population(cfg_pop)
    base_particle = pop.get_particle(pop.ids[0])
    cp = CoreShellParticle(base_particle, {"wvl_grid": [550e-9], "rh_grid": [0.0]})

    cp.h2o_vols = np.zeros_like(cp.rh_grid)
    cp.shell_dry_vol = 0.0
    for rr in range(len(cp.rh_grid)):
        for ww in range(len(cp.wvl_grid)):
            assert cp._shell_ri(rr, ww) == complex(1.0, 0.0)
