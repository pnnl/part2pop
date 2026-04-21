import numpy as np
from part2pop.analysis.population.factory.frac_ccn import build

def test_frac_ccn_between_zero_and_one(simple_population):
    cfg = {"s_grid": [0.1, 0.2], "T": 298.15}
    var = build(cfg)
    out = var.compute(simple_population, as_dict=True)
    frac = out["frac_ccn"]
    assert isinstance(frac, np.ndarray)
    assert frac.shape == (len(cfg["s_grid"]),)
    assert np.all(frac >= 0.0)
    assert np.all(frac <= 1.0 + 1e-12)
