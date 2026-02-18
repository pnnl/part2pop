import numpy as np

import part2pop.analysis.population.factory.avg_Jhet as jhet_mod


class _StubFreezePop:
    def __init__(self, value):
        self.value = value

    def get_avg_Jhet(self):
        return np.array([self.value])


def test_avg_Jhet_as_dict(monkeypatch):
    monkeypatch.setattr(
        jhet_mod, "build_freezing_population", lambda population, cfg: _StubFreezePop(0.5), raising=False
    )
    var = jhet_mod.build({"T_grid": [250.0]})
    arr = var.compute(population=None)
    assert np.allclose(arr, [0.5])
    out = var.compute(population=None, as_dict=True)
    assert out["avg_Jhet"].tolist() == [0.5]
