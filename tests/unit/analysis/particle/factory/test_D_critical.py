import numpy as np

from part2pop.analysis.particle.factory import D_critical
from .helpers import make_monodisperse_population


def test_d_critical_computes_values_against_population():
    pop = make_monodisperse_population(D_values=(90e-9, 150e-9), N_values=[1e4, 2e4])
    var = D_critical.build({})
    expected = [particle.get_critical_supersaturation(T=293.15, return_D_crit=True)[1] for particle in (pop.get_particle(pid) for pid in pop.ids)]
    assert np.allclose(var.compute_all(pop), expected)
    assert np.isclose(var.compute_one(pop, pop.ids[1]), expected[1])

    custom_T = 260.0
    var_custom = D_critical.build({"T": custom_T})
    expected_custom = [particle.get_critical_supersaturation(T=custom_T, return_D_crit=True)[1] for particle in (pop.get_particle(pid) for pid in pop.ids)]
    assert np.allclose(var_custom.compute_all(pop), expected_custom)
