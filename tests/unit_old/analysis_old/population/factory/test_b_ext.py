import numpy as np

from part2pop.analysis.population.factory.b_ext import ExtinctionCoeff, build


class DummyOpticalPopulation:
    def __init__(self, return_value):
        self.return_value = return_value

    def get_optical_coeff(self, name, rh=None, wvl=None):
        assert name == "b_ext"
        return self.return_value


def test_build_returns_extcoeff_instance():
    var = build({"rh_grid": [0.5], "wvl_grid": [450e-9], "morphology": "core-shell", "T": 298.15})
    assert isinstance(var, ExtinctionCoeff)
    assert var.meta.axis_names == ("rh_grid", "wvls")


def test_compute_handles_core_shell(simple_population, monkeypatch):
    cfg = {"rh_grid": [0.4], "wvl_grid": [355e-9], "morphology": "core-shell", "T": 300.0}
    fake_values = np.array([[2.5]])
    captured = {}

    def fake_builder(population, ocfg):
        captured["cfg"] = ocfg
        return DummyOpticalPopulation(fake_values)

    monkeypatch.setattr("part2pop.analysis.population.factory.b_ext.build_optical_population", fake_builder)

    var = build(cfg)
    out = var.compute(population=simple_population, as_dict=True)

    assert out["b_ext"].shape == (1, 1)
    assert np.allclose(out["b_ext"], fake_values)
    assert captured["cfg"]["type"] == "core_shell"
