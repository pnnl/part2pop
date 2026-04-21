import numpy as np

from part2pop.analysis.population.factory.b_scat import BScatVar, build


class DummyOpticalPopulation:
    def __init__(self, return_value):
        self.return_value = return_value

    def get_optical_coeff(self, name, rh=None, wvl=None):
        assert name == "b_scat"
        return self.return_value


def test_build_returns_bscat_instance():
    var = build({"rh_grid": [0.3], "wvls": [532e-9], "morphology": "core-shell", "T": 290.0})
    assert isinstance(var, BScatVar)
    assert var.meta.name == "b_scat"


def test_compute_returns_array(simple_population, monkeypatch):
    cfg = {"rh_grid": [0.6], "wvls": [550e-9], "morphology": "core-shell", "T": 285.0}
    fake_values = np.array([[0.42]])
    captured = {}

    def fake_builder(population, ocfg):
        captured["cfg"] = ocfg
        return DummyOpticalPopulation(fake_values)

    monkeypatch.setattr("part2pop.analysis.population.factory.b_scat.build_optical_population", fake_builder)

    var = build(cfg)
    out = var.compute(population=simple_population, as_dict=True)

    assert np.allclose(out["b_scat"], fake_values)
    assert out["wvl_grid"].shape == (1,)
    assert captured["cfg"]["wvl_grid"] == cfg["wvls"]
