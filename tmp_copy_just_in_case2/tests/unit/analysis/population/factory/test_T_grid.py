import numpy as np
import pytest

import part2pop.analysis.population.factory.T_grid as T_grid_mod


def test_T_grid_converts_units_and_errors():
    var = T_grid_mod.build({"T_grid": [0.0, 25.0], "T_units": "C"})
    vals = var.compute()
    assert np.allclose(vals, [273.15, 298.15])

    with pytest.raises(ValueError):
        T_grid_mod.build({"T_grid": [1.0], "T_units": "X"}).compute()
