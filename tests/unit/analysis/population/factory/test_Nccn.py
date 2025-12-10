import numpy as np

import part2pop.analysis.population.factory.Nccn as Nccn_mod


class _StubParticle:
    def __init__(self, scrit):
        self._scrit = scrit

    def get_critical_supersaturation(self, T, return_D_crit=False):
        return self._scrit


class _StubPopulation:
    def __init__(self, scrits, num_concs):
        self.ids = list(range(len(scrits)))
        self.num_concs = np.asarray(num_concs, dtype=float)
        self._parts = {pid: _StubParticle(scrit) for pid, scrit in zip(self.ids, scrits)}

    def get_particle(self, pid):
        return self._parts[pid]


def test_nccn_counts_above_threshold():
    pop = _StubPopulation(scrits=[0.1, 0.2, 0.3], num_concs=[1.0, 2.0, 3.0])
    var = Nccn_mod.build({"T": 298.15, "s_grid": [0.0, 0.15, 0.25, 0.35]})

    nccn = var.compute(pop)
    # At each s, count particles with s_crit <= s
    assert np.allclose(nccn, [0.0, 1.0, 3.0, 6.0])

    as_dict = var.compute(pop, as_dict=True)
    assert "s" in as_dict and "Nccn" in as_dict
