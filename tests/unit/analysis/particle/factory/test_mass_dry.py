import numpy as np

from part2pop.analysis.particle.factory import mass_dry
from .helpers import make_monodisperse_population


def test_mass_dry_computes_values_against_population():
    pop = make_monodisperse_population(D_values=(90e-9, 150e-9))
    var = mass_dry.build({})

    expected = [pop.get_particle(pid).get_mass_dry() for pid in pop.ids]

    assert np.allclose(var.compute_all(pop), expected)
    assert np.isclose(var.compute_one(pop, pop.ids[1]), expected[1])
