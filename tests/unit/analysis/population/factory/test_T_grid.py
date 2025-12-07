import numpy as np
import pytest

from part2pop.analysis.population.factory.T_grid import TemperatureGridVar, build


def test_build_returns_temperature_grid_var():
    var = build({"T_grid": [250.0]})
    assert isinstance(var, TemperatureGridVar)
    assert var.meta.units == "K"


def test_compute_converts_celsius_to_kelvin():
    var = build({"T_grid": [0.0, 10.0], "T_units": "C"})
    out = var.compute()
    assert np.allclose(out, np.array([273.15, 283.15]))


def test_compute_raises_on_unknown_units():
    var = build({"T_grid": [1.0], "T_units": "F"})
    with pytest.raises(ValueError):
        var.compute()
