import numpy as np
import pytest

import part2pop.analysis.population.factory.frac_ccn as frac_mod


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


def test_frac_ccn_warns_on_mismatched_grids():
    pop = _StubPopulation(scrits=[0.2, 0.4], num_concs=[2.0, 2.0])
    var = frac_mod.build({"T": 298.15, "s_grid": [0.1, 0.5], "s_eval": [0.1]})

    with pytest.warns(UserWarning):
        frac = var.compute(pop)

    # At s=0.1 none activate; at s=0.5 both activate -> 0 then 1
    assert np.allclose(frac, [0.0, 1.0])

    as_dict = var.compute(pop, as_dict=True)
    assert set(as_dict) == {"s", "frac_ccn"}
