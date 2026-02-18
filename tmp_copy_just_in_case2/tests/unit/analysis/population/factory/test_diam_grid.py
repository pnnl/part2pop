import numpy as np

import part2pop.analysis.population.factory.diam_grid as diam_mod


def test_diam_grid_as_dict():
    var = diam_mod.build({"diam_grid": [1, 2, 3]})
    arr = var.compute(None)
    assert arr.tolist() == [1, 2, 3]
    out = var.compute(None, as_dict=True)
    assert "diam_grid" in out
