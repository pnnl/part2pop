import numpy as np
import pytest

import part2pop.analysis.population.factory.dNdlnD as dnd_mod


class _StubParticle:
    def __init__(self, dwet, ddry):
        self._dwet = dwet
        self._ddry = ddry

    def get_Dwet(self):
        return self._dwet

    def get_Ddry(self):
        return self._ddry


class _StubPopulation:
    def __init__(self, dwets, ddrys, num_concs):
        self.ids = list(range(len(dwets)))
        self.num_concs = np.asarray(num_concs, dtype=float)
        self._parts = {
            pid: _StubParticle(dw, dr) for pid, dw, dr in zip(self.ids, dwets, ddrys)
        }

    def get_particle(self, pid):
        return self._parts[pid]


def test_dnd_hist_returns_density_and_as_dict():
    pop = _StubPopulation([1e-7, 2e-7, 3e-7], [0.5e-7, 1e-7, 1.5e-7], [1.0, 2.0, 3.0])
    var = dnd_mod.build({"method": "hist", "N_bins": 2, "D_min": 1e-7, "D_max": 3e-7})

    dens = var.compute(pop)
    assert dens.shape[0] == 2
    as_dict = var.compute(pop, as_dict=True)
    assert set(as_dict) == {"D", "dNdlnD", "edges"}


def test_dnd_uses_dry_diam_and_errors_on_bad_edges():
    pop = _StubPopulation([1e-7, 2e-7], [1e-8, 2e-8], [1.0, 1.0])
    var = dnd_mod.build({"method": "hist", "wetsize": False, "edges": [1e-9, 1e-8, 1e-7]})
    dens = var.compute(pop)
    assert dens.shape[0] == 2

    var_bad = dnd_mod.build({"method": "hist", "edges": [0.0, 1.0]})
    with pytest.raises(ValueError):
        var_bad.compute(pop)


def test_dnd_kde_requires_grid(monkeypatch):
    pop = _StubPopulation([1e-7], [1e-7], [1.0])
    var = dnd_mod.build({"method": "kde", "diam_grid": [1e-7, 2e-7], "normalize": True})
    with pytest.raises(ValueError):
        var.compute(pop)

    # unknown method should raise
    bad = dnd_mod.build({"method": "unknown"})
    with pytest.raises(ValueError):
        bad.compute(pop)
