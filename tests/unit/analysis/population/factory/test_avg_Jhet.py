import numpy as np

from part2pop.analysis.population.factory.avg_Jhet import avgJhetVar, build


class DummyFreezingPopulation:
    def __init__(self, return_value):
        self.return_value = return_value

    def get_avg_Jhet(self):
        return self.return_value


def test_build_returns_avg_jhet_var():
    var = build({"T_grid": [250.0]})
    assert isinstance(var, avgJhetVar)
    assert "T_grid" in var.meta.axis_names


def test_compute_calls_freezing_builder(monkeypatch):
    fake = DummyFreezingPopulation(np.array([1.0, 2.0]))

    def fake_builder(population, cfg):
        assert "T_grid" in cfg
        return fake

    monkeypatch.setattr("part2pop.analysis.population.factory.avg_Jhet.build_freezing_population", fake_builder)

    var = build({"T_grid": [250.0, 260.0], "morphology": "homogeneous"})
    out = var.compute(population=object(), as_dict=True)

    assert np.allclose(out["avg_Jhet"], fake.return_value)
    assert out["T_units"] == "K"
