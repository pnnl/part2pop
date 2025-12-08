import numpy as np

import part2pop.analysis.population.factory.rh_grid as rh_mod


def test_rh_grid_defaults_and_override():
    assert rh_mod.build({}).compute(None).tolist() == [0.0]
    var = rh_mod.build({"rh_grid": [0.1, 0.2]})
    assert var.compute(None, as_dict=True)["rh_grid"].tolist() == [0.1, 0.2]
