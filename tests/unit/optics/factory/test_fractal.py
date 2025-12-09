# tests/unit/optics/factory/test_fractal.py

import importlib
import sys
import types

import numpy as np
import pytest

from part2pop.population.builder import build_population
from part2pop.optics.factory.fractal import FractalParticle, build


def _make_bc_rich_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.5, 0.4, 0.1]],
    }
    return build_population(cfg)


# @pytest.mark.importorskip("pyBCabs")
def test_fractal_particle_compute_optics():
    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])

    cfg = {"wvl_grid": [550e-9], "rh_grid": [0.0]}
    fp = FractalParticle(base_particle, cfg)
    fp.compute_optics()

    assert np.isfinite(fp.Cext[0, 0])
    assert fp.Cext[0, 0] >= 0.0


@pytest.mark.importorskip("pyBCabs")
def test_fractal_build_wrapper():
    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])
    fp = build(base_particle, {"wvl_grid": [550e-9], "rh_grid": [0.0]})
    assert isinstance(fp, FractalParticle)


def test_fractal_particle_with_stubbed_dependencies(monkeypatch):
    fake_pbca = types.SimpleNamespace(
        small_PSP=lambda *a, **k: 1.0,
        large_PSP=lambda *a, **k: 2.0,
    )
    fake_pkg = types.ModuleType("pyBCabs")
    fake_pkg.retrieval = fake_pbca
    monkeypatch.setitem(sys.modules, "pyBCabs", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyBCabs.retrieval", fake_pbca)

    fake_pymie = types.ModuleType("PyMieScatt")
    fake_pymie.MieQCoreShell = lambda *a, **k: {"Qsca": 1.0, "g": 0.5}
    monkeypatch.setitem(sys.modules, "PyMieScatt", fake_pymie)

    def stub_build_ri(spec, wvl_grid, modifications=None):
        spec.refractive_index = types.SimpleNamespace(
            real_ri_fun=lambda w: np.ones_like(w),
            imag_ri_fun=lambda w: np.zeros_like(w),
        )

    monkeypatch.setattr("part2pop.optics.refractive_index.build_refractive_index", stub_build_ri)

    sys.modules.pop("part2pop.optics.factory.fractal", None)
    fractal = importlib.import_module("part2pop.optics.factory.fractal")

    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])
    fp = fractal.FractalParticle(
        base_particle,
        {
            "wvl_grid": [550e-9],
            "rh_grid": [0.0, 0.5],
            "temp": 298.15,
            "single_scatter_albedo": 0.8,
        },
    )

    assert fp.Cabs.shape == (2, 1)
    assert isinstance(fp._shell_ri(0, 0), complex)
    assert fp._shell_ri(1, 0).real >= 0.0
    x_core, x_coated = fp.get_x(1.0, 1.0)
    assert isinstance(x_core, float)
    assert isinstance(x_coated, float)
    core_Df, coated_Df = fp.get_Df(10.0, 1.5)
    assert core_Df != coated_Df
    assert isinstance(fp.get_phase_shift(1.0, core_Df, 500e-9), float)
