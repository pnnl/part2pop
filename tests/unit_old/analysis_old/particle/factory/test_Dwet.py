import numpy as np

from part2pop.analysis.particle.factory.Dwet import Dwet, build


def test_build_returns_dwet_variable():
    var = build({})
    assert isinstance(var, Dwet)
    assert var.meta.units == "m"


def test_compute_methods(simple_population):
    pop = simple_population
    var = build({})

    assert np.isclose(var.compute_one(pop, 0), pop.get_particle(0).get_Dwet())
    assert np.allclose(var.compute_all(pop), np.array([p.get_Dwet() for p in map(pop.get_particle, pop.ids)]))
