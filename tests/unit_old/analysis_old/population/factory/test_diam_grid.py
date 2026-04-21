import numpy as np

from part2pop.analysis.population.factory.diam_grid import DiamGridVar, build


def test_build_returns_diam_grid_var():
    var = build({"diam_grid": [1e-9, 2e-9]})
    assert isinstance(var, DiamGridVar)
    assert var.meta.scale == "log"


def test_compute_returns_numpy_array():
    cfg = {"diam_grid": [1e-9, 2e-9, 3e-9]}
    var = build(cfg)
    out = var.compute(population=None, as_dict=True)
    assert np.allclose(out["diam_grid"], np.array(cfg["diam_grid"]))
