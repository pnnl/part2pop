import numpy as np

import part2pop.analysis.population.factory.b_scat as b_scat_mod


class _StubOpticalPop:
    def __init__(self, value):
        self.value = value

    def get_optical_coeff(self, name, rh=None, wvl=None):
        return np.full((2, 2), self.value)


def test_b_scat_compute_and_as_dict(monkeypatch):
    monkeypatch.setattr(b_scat_mod, "build_optical_population", lambda pop, cfg: _StubOpticalPop(3.45))
    cfg = {"rh_grid": [0.0, 0.5], "wvl_grid": [500e-9, 600e-9], "T": 298.15, "morphology": "core-shell"}
    var = b_scat_mod.build(cfg)
    arr = var.compute(population=None)
    assert arr.shape == (2, 2)
    out = var.compute(population=None, as_dict=True)
    assert "rh_grid" in out and "wvl_grid" in out
