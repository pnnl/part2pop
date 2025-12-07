import numpy as np

from part2pop.analysis.population.factory.s_grid import SupersaturationGridVar, build


def test_build_returns_supersaturation_grid():
    var = build({"s_grid": [0.1, 0.2]})
    assert isinstance(var, SupersaturationGridVar)
    assert var.meta.units == "%"


def test_compute_prefers_s_grid_over_s_eval():
    cfg = {"s_grid": [0.1], "s_eval": [0.3, 0.4]}
    var = build(cfg)
    out = var.compute(as_dict=True)

    assert np.allclose(out["s_grid"], np.array([0.1]))
