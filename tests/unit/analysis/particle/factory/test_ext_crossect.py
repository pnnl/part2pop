import numpy as np

from part2pop.analysis.particle.factory import ext_crossect
from .helpers import make_monodisperse_population


def _build_optical_pop(values, attr_name):
    class _Pop:
        def __init__(self, values):
            setattr(self, attr_name, np.asarray(values, dtype=float).reshape(-1, 1, 1))

    return _Pop(values)


def test_ext_crossect_builds_core_shell_cfg(monkeypatch):
    captured = {}
    def fake_build(population, cfg):
        captured["cfg"] = cfg
        return _build_optical_pop([1.0, 2.0], "Cext")

    monkeypatch.setattr(ext_crossect, "build_optical_population", fake_build)
    pop = make_monodisperse_population(D_values=(100e-9, 150e-9))

    var = ext_crossect.build({"RH": 0.5, "wvl": 600e-9, "T": 300.0, "morphology": "core-shell"})
    assert np.allclose(var.compute_all(pop), [1.0, 2.0])
    assert np.isclose(var.compute_one(pop, pop.ids[-1]), 2.0)

    cfg = captured["cfg"]
    assert cfg["type"] == "core_shell"
    assert cfg["rh_grid"] == [0.5]
    assert cfg["wvl_grid"] == [600e-9]
    assert np.isclose(cfg["temp"], 300.0)


def test_ext_crossect_respects_predefined_grid(monkeypatch):
    captured = {}
    def fake_build(population, cfg):
        captured["cfg"] = cfg
        return _build_optical_pop([3.0], "Cext")

    monkeypatch.setattr(ext_crossect, "build_optical_population", fake_build)
    pop = make_monodisperse_population(D_values=(110e-9,))
    cfg = {"rh_grid": [0.2], "wvl_grid": [500e-9], "T": 280.0}
    var = ext_crossect.build(cfg)
    assert np.allclose(var.compute_all(pop), [3.0])

    built_cfg = captured["cfg"]
    assert built_cfg["rh_grid"] == [0.2]
    assert built_cfg["wvl_grid"] == [500e-9]
    assert np.isclose(built_cfg["temp"], 280.0)
