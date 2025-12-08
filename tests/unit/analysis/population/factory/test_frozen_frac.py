import numpy as np

import part2pop.analysis.population.factory.frozen_frac as ff_mod


class _StubFreezePop:
    def __init__(self, value):
        self.value = value

    def get_frozen_fraction(self, cooling_rate):
        return np.array([self.value + cooling_rate])


def test_frozen_frac_uses_cooling_rate(monkeypatch):
    monkeypatch.setattr(
        ff_mod, "build_freezing_population", lambda population, cfg: _StubFreezePop(0.2), raising=False
    )
    var = ff_mod.build({"T_grid": [250.0], "cooling_rate": 1.0})
    arr = var.compute(population=None)
    assert np.allclose(arr, [1.2])  # base 0.2 + cooling_rate
