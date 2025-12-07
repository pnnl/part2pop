import numpy as np

from part2pop.analysis.population.factory.nucleating_sites import NucleatingSites, build


class DummyFreezingPopulation:
    def __init__(self, return_value):
        self.return_value = return_value

    def get_nucleating_sites(self, cooling_rate):
        assert cooling_rate == 5.0
        return self.return_value


def test_build_returns_nucleating_sites_var():
    var = build({"T_grid": [230.0], "cooling_rate": 5.0})
    assert isinstance(var, NucleatingSites)
    assert "T_grid" in var.meta.axis_names


def test_compute_calls_freezing_population(monkeypatch):
    fake = DummyFreezingPopulation(np.array([1e5, 2e5]))

    def fake_builder(population, freezing_cfg):
        assert "T_grid" in freezing_cfg
        return fake

    monkeypatch.setattr("part2pop.analysis.population.factory.nucleating_sites.build_freezing_population", fake_builder)

    cfg = {"T_grid": [230.0, 240.0], "cooling_rate": 5.0}
    var = build(cfg)
    out = var.compute(population=object(), as_dict=True)

    assert np.allclose(out["nucleating_sites"], fake.return_value)
    assert out["cooling_rate"] == cfg["cooling_rate"]
