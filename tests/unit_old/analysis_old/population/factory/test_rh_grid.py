import numpy as np

from part2pop.analysis.population.factory.rh_grid import RHGridVar, build


def test_build_returns_rh_grid_var():
    var = build({"rh_grid": [0.3, 0.6]})
    assert isinstance(var, RHGridVar)
    assert var.meta.short_label == "RH"


def test_compute_returns_array():
    cfg = {"rh_grid": [0.1, 0.2]}
    var = build(cfg)
    out = var.compute(as_dict=True)
    assert np.allclose(out["rh_grid"], np.array(cfg["rh_grid"]))
