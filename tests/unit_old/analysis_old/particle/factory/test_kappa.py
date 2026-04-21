import numpy as np

from part2pop.analysis.particle.factory.kappa import kappa, build


def test_build_returns_kappa_variable():
    var = build({})
    assert isinstance(var, kappa)
    assert var.meta.name == "kappa"


def test_compute_methods_return_expected_values(simple_population):
    pop = simple_population
    var = build({})

    one = var.compute_one(pop, part_id=0)
    assert np.isclose(one, 0.3)

    all_vals = var.compute_all(pop)
    assert np.allclose(all_vals, np.array([0.3, 0.5]))
