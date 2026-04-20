import numpy as np

from part2pop.analysis.particle.factory import scat_crossect


from .helpers import make_monodisperse_population

def _build_scatter_pop(values):
    class _Pop:
        def __init__(self, values):
            self.Csca = np.asarray(values, dtype=float).reshape(-1, 1, 1)

    return _Pop(values)


def test_scat_crossect_builds_precision_cfg(monkeypatch):
    calls = {}
    def fake_build(population, cfg):
        calls["cfg"] = cfg
        return _build_scatter_pop([7.0, 8.0])

    monkeypatch.setattr(scat_crossect, "build_optical_population", fake_build)
    pop = make_monodisperse_population(D_values=(95e-9, 120e-9))

    var = scat_crossect.build({"RH": 0.4, "wvl": 500e-9, "morphology": "core-shell"})
    assert np.allclose(var.compute_all(pop), [7.0, 8.0])
    assert np.isclose(var.compute_one(pop, pop.ids[0]), 7.0)

    cfg = calls["cfg"]
    assert cfg["rh_grid"] == [0.4]
    assert cfg["wvl_grid"] == [500e-9]


def test_scat_crossect_uses_grids_when_provided(monkeypatch):
    calls = {}
    def fake_build(population, cfg):
        calls["cfg"] = cfg
        return _build_scatter_pop([9.0])

    monkeypatch.setattr(scat_crossect, "build_optical_population", fake_build)
    pop = make_monodisperse_population(D_values=(105e-9,))
    var = scat_crossect.build({"rh_grid": [0.3], "wvl_grid": [450e-9], "T": 290.0})

    assert np.allclose(var.compute_all(pop), [9.0])
    built_cfg = calls["cfg"]
    assert built_cfg["rh_grid"] == [0.3]
    assert built_cfg["wvl_grid"] == [450e-9]
    assert np.isclose(built_cfg["temp"], 290.0)