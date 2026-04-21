import numpy as np

from part2pop.analysis.population.factory.frozen_frac import FrozenFraction, build


class DummyFreezingPopulation:
    def __init__(self, expected_rate):
        self.expected_rate = expected_rate

    def get_frozen_fraction(self, cooling_rate):
        assert cooling_rate == self.expected_rate
        return np.array([0.1, 0.2])


def test_build_returns_frozen_fraction_var():
    var = build({"T_grid": [250.0], "cooling_rate": 5.0})
    assert isinstance(var, FrozenFraction)
    assert "T_grid" in var.meta.axis_names


def test_compute_calls_freezing_population(monkeypatch):
    cfg = {"T_grid": [240.0, 245.0], "cooling_rate": 3.0}
    fake = DummyFreezingPopulation(expected_rate=cfg["cooling_rate"])

    def fake_builder(population, freezing_cfg):
        assert freezing_cfg["T_grid"] == cfg["T_grid"]
        return fake

    monkeypatch.setattr("part2pop.analysis.population.factory.frozen_frac.build_freezing_population", fake_builder)

    var = build(cfg)
    out = var.compute(population=object(), as_dict=True)

    assert np.allclose(out["nucleating_sites"], fake.get_frozen_fraction(cfg["cooling_rate"]))
    assert out["cooling_rate"] == cfg["cooling_rate"]
