import numpy as np

from part2pop.analysis.particle.factory import s_critical


class _DummyParticle:
    def __init__(self, idx, base=0.1):
        self.idx = idx
        self.base = base

    def get_critical_supersaturation(self, T=None, return_D_crit=False):
        value = self.base * self.idx
        if return_D_crit:
            return value, value * 2
        return value


class _DummyPopulation:
    def __init__(self, ids):
        self.ids = list(ids)

    def get_particle(self, part_id):
        return _DummyParticle(part_id)


def test_s_critical_defaults_and_configs():
    pop = _DummyPopulation(ids=(2, 5))
    var = s_critical.build({})

    assert np.allclose(var.compute_all(pop), [0.2, 0.5])
    assert np.isclose(var.compute_one(pop, 5), 0.5)

    var_custom = s_critical.build({"T": 260.0})
    assert np.allclose(var_custom.compute_all(pop), [0.2, 0.5])
    assert np.isclose(var_custom.compute_from_particle(pop.get_particle(2)), 0.2)