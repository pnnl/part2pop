import pytest

from part2pop.analysis.particle.factory import registry
from part2pop.analysis.particle.factory.Dwet import Dwet


def test_list_particle_variables_discovers_factories():
    names = registry.list_particle_variables()
    assert "Dwet" in names
    assert "kappa" in names


def test_alias_resolution_returns_builder():
    builder = registry.get_particle_builder("wet_diameter")
    inst = builder({})
    assert isinstance(inst, Dwet)


def test_unknown_particle_variable_raises():
    with pytest.raises(registry.UnknownParticleVariableError):
        registry.get_particle_builder("not_a_particle_var")
