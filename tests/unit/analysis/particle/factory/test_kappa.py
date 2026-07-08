import numpy as np

from part2pop.analysis.particle.factory import kappa
from .helpers import make_monodisperse_population


def test_kappa_computes_values_against_population():
    pop = make_monodisperse_population(D_values=(90e-9, 150e-9))
    var = kappa.build({})

    expected = [pop.get_particle(pid).get_tkappa() for pid in pop.ids]

    assert np.allclose(var.compute_all(pop), expected)
    assert np.isclose(var.compute_one(pop, pop.ids[0]), expected[0])
