import numpy as np

import part2pop.analysis.population.factory.wvl_grid as wvl_mod


def test_wvl_grid_handles_sequence():
    var = wvl_mod.build({"wvl_grid": [550e-9, 650e-9]})
    arr = var.compute(None)
    assert np.allclose(arr, [550e-9, 650e-9])
    assert "wvl_grid" in var.compute(None, as_dict=True)
