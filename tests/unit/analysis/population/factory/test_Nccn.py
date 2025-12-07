import numpy as np

from part2pop.analysis.population.factory.Nccn import NccnVar, build


def test_build_returns_nccn_var():
    var = build({"s_grid": [0.1], "T": 298.15})
    assert isinstance(var, NccnVar)
    assert var.meta.axis_names == ("s",)


def test_compute_counts(simple_population):
    cfg = {"s_grid": [0.1, 0.2], "T": 298.15}
    var = build(cfg)
    out = var.compute(simple_population, as_dict=True)

    assert np.allclose(out["Nccn"], np.array([200.0, 300.0]))
    assert np.allclose(out["s"], np.array(cfg["s_grid"]))
