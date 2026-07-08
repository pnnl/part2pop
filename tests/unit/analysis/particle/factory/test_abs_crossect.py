import numpy as np

from part2pop.analysis.particle.factory import abs_crossect
from .helpers import make_monodisperse_population


def _build_abs_pop(values):
    class _Pop:
        def __init__(self, values):
            self.Cabs = np.asarray(values, dtype=float).reshape(-1, 1, 1)

    return _Pop(values)


def test_abs_crossect_builds_core_shell_cfg(monkeypatch):
    calls = {}

    def fake_build(population, cfg):
        calls["cfg"] = cfg
        return _build_abs_pop([1.0, 2.0])

    monkeypatch.setattr(abs_crossect, "build_optical_population", fake_build)
    pop = make_monodisperse_population(D_values=(95e-9, 120e-9))

    var = abs_crossect.build({"RH": 0.4, "wvl": 500e-9, "morphology": "core-shell", "T": 290.0})

    assert np.allclose(var.compute_all(pop), [1.0, 2.0])
    assert np.isclose(var.compute_one(pop, pop.ids[1]), 2.0)
    assert calls["cfg"]["type"] == "core_shell"
    assert calls["cfg"]["rh_grid"] == [0.4]
    assert calls["cfg"]["wvl_grid"] == [500e-9]
    assert np.isclose(calls["cfg"]["temp"], 290.0)
