import numpy as np

import part2pop.analysis.population.factory.nucleating_sites as ns_mod


class _StubFreezePop:
    def __init__(self, value):
        self.value = value

    def get_nucleating_sites(self, cooling_rate):
        return np.array([self.value * cooling_rate])


def test_nucleating_sites(monkeypatch):
    monkeypatch.setattr(
        ns_mod, "build_freezing_population", lambda population, cfg: _StubFreezePop(2.0), raising=False
    )
    var = ns_mod.build({"T_grid": [250.0], "cooling_rate": 0.5})
    arr = var.compute(population=None)
    assert np.allclose(arr, [1.0])
