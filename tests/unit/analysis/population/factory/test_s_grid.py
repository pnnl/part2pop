import numpy as np

import part2pop.analysis.population.factory.s_grid as s_mod


def test_s_grid_prefers_s_grid():
    var = s_mod.build({"s_grid": [0.01, 0.02], "s_eval": [0.3]})
    vals = var.compute(None)
    assert vals.tolist() == [0.01, 0.02]
    as_dict = var.compute(None, as_dict=True)
    assert as_dict["s_grid"].tolist() == [0.01, 0.02]
