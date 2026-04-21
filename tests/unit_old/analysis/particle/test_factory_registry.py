from part2pop.analysis.particle.factory import registry
from part2pop.analysis.particle.base import ParticleVariable

def test_particle_registry_lists_core_variables():
    names = registry.list_variables()
    for expected in ("kappa", "Dwet", "P_frz"):
        assert expected in names

def test_particle_registry_get_builder():
    builder = registry.get_variable_builder("kappa")
    var = builder({})
    assert isinstance(var, ParticleVariable)
