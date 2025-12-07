import numpy as np
import pytest

from part2pop.analysis.particle.factory.P_frz import FreezingProb, build


class DummyFreezingPopulation:
    def __init__(self, values):
        self.values = np.asarray(values)

    def get_freezing_probs(self):
        return self.values


def test_build_returns_freezing_prob_var():
    var = build({"T": 250.0})
    assert isinstance(var, FreezingProb)
    assert var.meta.name == "P_frz"


def test_compute_raises_without_temperature():
    var = build({})
    with pytest.raises(ValueError):
        var.compute_all(population=object())


def test_compute_calls_freezing_population(monkeypatch):
    fake = DummyFreezingPopulation([0.1, 0.2])

    def fake_builder(population, freezing_cfg):
        assert freezing_cfg["T_grid"] == [255.0]
        return fake

    monkeypatch.setattr("part2pop.analysis.particle.factory.P_frz.build_freezing_population", fake_builder)

    var = build({"T": 255.0})
    out = var.compute_all(population=object())

    assert np.isclose(out, 0.1)
