import numpy as np
import pytest

import part2pop.analysis.particle.factory.Dwet as Dwet
import part2pop.analysis.particle.factory.kappa as kappa
import part2pop.analysis.particle.factory.P_frz as P_frz


class _StubParticle:
    def __init__(self, d=1.0, k=0.3):
        self._d = d
        self._k = k

    def get_Dwet(self):
        return self._d

    def get_tkappa(self):
        return self._k


class _StubPopulation:
    def __init__(self, ids=(1, 2)):
        self.ids = ids
        self._particles = {pid: _StubParticle(d=pid, k=pid * 0.1) for pid in ids}

    def get_particle(self, part_id):
        return self._particles[part_id]


def test_dwet_variable_computes(monkeypatch):
    pop = _StubPopulation(ids=(1, 2))
    var = Dwet.build({})

    assert var.compute_one(pop, 1) == 1
    all_vals = var.compute_all(pop)
    assert np.allclose(all_vals, [1, 2])


def test_kappa_variable_computes(monkeypatch):
    pop = _StubPopulation(ids=(3, 4))
    var = kappa.build({})

    assert np.isclose(var.compute_one(pop, 3), 0.3)
    all_vals = var.compute_all(pop)
    assert np.allclose(all_vals, [0.3, 0.4])


def test_p_frz_requires_temperature_and_uses_builder(monkeypatch):
    pop = _StubPopulation(ids=(1,))

    # Missing T should raise
    var = P_frz.build({"T": None})
    with pytest.raises(ValueError):
        var.compute_all(pop)

    # Stub out build_freezing_population to return object with get_freezing_probs
    class _StubFreezePop:
        def get_freezing_probs(self):
            return np.array([[0.42]])

    monkeypatch.setattr(
        P_frz, "build_freezing_population", lambda population, cfg: _StubFreezePop()
    )

    var_with_T = P_frz.build({"T": 250.0})
    probs = var_with_T.compute_all(pop)
    assert np.allclose(probs, 0.42)
