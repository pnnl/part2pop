import numpy as np

from part2pop.analysis.population.factory.frac_ccn import FracCCNVar, build


def test_build_returns_frac_ccn_var():
    var = build({"s_grid": [0.1], "T": 298.15})
    assert isinstance(var, FracCCNVar)
    assert var.meta.name == "frac_ccn"


def test_compute_fraction(simple_population):
    cfg = {"s_grid": [0.1, 0.2], "T": 298.15}
    var = build(cfg)
    out = var.compute(simple_population, as_dict=True)

    assert np.allclose(out["frac_ccn"], np.array([200.0 / 300.0, 1.0]))
    assert out["s"].shape == (2,)
