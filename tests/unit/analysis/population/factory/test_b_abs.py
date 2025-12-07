import numpy as np

from part2pop.analysis.population.factory.b_abs import AbsCoeff, build


class DummyOpticalPopulation:
    def __init__(self, expected_cfg, return_value):
        self.expected_cfg = expected_cfg
        self.return_value = return_value

    def get_optical_coeff(self, name, rh=None, wvl=None):
        assert name == "b_abs"
        return self.return_value


def test_build_returns_abscoeff_instance():
    var = build({"rh_grid": [0.5], "wvls": [550e-9], "morphology": "core-shell", "T": 280.0})
    assert isinstance(var, AbsCoeff)
    assert var.meta.name == "b_abs"
    assert var.meta.axis_names == ("rh_grid", "wvls")


def test_compute_uses_optical_population(simple_population, monkeypatch):
    cfg = {"rh_grid": [0.5], "wvls": [550e-9], "morphology": "core-shell", "T": 275.0}
    captured = {}

    def fake_builder(population, ocfg):
        captured["cfg"] = ocfg
        return DummyOpticalPopulation(ocfg, np.array([[1.23]]))

    monkeypatch.setattr("part2pop.analysis.population.factory.b_abs.build_optical_population", fake_builder)

    var = build(cfg)
    out = var.compute(population=simple_population)

    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1)
    assert captured["cfg"]["type"] == "core_shell"
