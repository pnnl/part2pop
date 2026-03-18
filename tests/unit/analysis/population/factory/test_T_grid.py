import numpy as np
import pytest

import part2pop.analysis.population.factory.T_grid as T_grid_mod


def test_T_grid_converts_units_and_errors():
    var = T_grid_mod.build({"T_grid": [0.0, 25.0], "T_units": "C"})
    vals = var.compute()
    assert np.allclose(vals, [0.0, 25.0])
    assert var.cfg['T_units']=="C"

    var = T_grid_mod.build({"T_grid": [273.0, 298.0], "T_units": "K"})
    vals = var.compute()
    assert np.allclose(vals, [273.0, 298.0])
    assert var.cfg['T_units']=="K"

    with pytest.raises(ValueError):
        T_grid_mod.build({"T_grid": [1.0], "T_units": "X"}).compute()

    with pytest.raises(ValueError):
        T_grid_mod.build({"T_grid": [-10.0], "T_units": "K"}).compute()
