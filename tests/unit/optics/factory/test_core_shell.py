# tests/unit/optics/factory/test_core_shell.py

import importlib
import sys
import types

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


def test_core_shell_with_stubbed_pymie(monkeypatch):
    fake_pymie = types.ModuleType("PyMieScatt")

    def fake_mieq(*args, **kwargs):
        return {"Qext": 1.0, "Qsca": 0.5, "Qabs": 0.3, "g": 0.4}

    fake_pymie.MieQCoreShell = fake_mieq
    monkeypatch.setitem(sys.modules, "PyMieScatt", fake_pymie)

    def stub_build_ri(spec, wvl_grid, modifications=None):
        spec.refractive_index = types.SimpleNamespace(
            real_ri_fun=lambda w: np.full_like(w, 1.5),
            imag_ri_fun=lambda w: np.zeros_like(w),
        )

    monkeypatch.setattr(
        "part2pop.optics.refractive_index.build_refractive_index", stub_build_ri
    )

    core_shell_mod = importlib.reload(
        importlib.import_module("part2pop.optics.factory.core_shell")
    )

    cfg_pop = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.2, 0.7, 0.1]],
    }
    pop = build_population(cfg_pop)
    base_particle = pop.get_particle(pop.ids[0])
    cfg = {"wvl_grid": [550e-9], "rh_grid": [0.0, 0.5]}
    cp = core_shell_mod.CoreShellParticle(base_particle, cfg)
    assert cp.Cext.shape == (2, 1)
    assert cp._shell_ri(0, 0).real >= 0.0
