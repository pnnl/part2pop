import numpy as np

from part2pop.analysis.particle.factory import SSA
from .helpers import make_monodisperse_population


class _OptPop:
    def __init__(self, csca, cext):
        self.Csca = np.asarray(csca, dtype=float).reshape(-1, 1, 1)
        self.Cext = np.asarray(cext, dtype=float).reshape(-1, 1, 1)


def test_ssa_computes_ratio(monkeypatch):
    pop = make_monodisperse_population(D_values=(95e-9, 120e-9))

    def fake_build(population, cfg):
        return _OptPop([3.0, 4.0], [5.0, 8.0])

    monkeypatch.setattr(SSA, "build_optical_population", fake_build)
    var = SSA.build({"RH": 0.4, "wvl": 500e-9})

    assert np.allclose(var.compute_all(pop), [3.0 / 5.0, 4.0 / 8.0])
    assert np.isclose(var.compute_one(pop, pop.ids[0]), 3.0 / 5.0)


def test_ssa_returns_zero_when_extinction_is_zero(monkeypatch):
    pop = make_monodisperse_population(D_values=(95e-9, 120e-9))

    def fake_build(population, cfg):
        return _OptPop([2.0, 0.0], [0.0, 0.0])

    monkeypatch.setattr(SSA, "build_optical_population", fake_build)
    var = SSA.build({})

    assert np.allclose(var.compute_all(pop), [0.0, 0.0])
