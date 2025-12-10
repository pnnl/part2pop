import pytest

from part2pop.analysis.particle.factory import registry as reg


def test_particle_registry_exposes_known_variables():
    names = reg.list_particle_variables(include_aliases=True)
    assert "Dwet" in names
    assert "wet_diameter" in names
    assert "kappa" in names  # alias for the hygroscopicity variable


def test_resolve_alias_returns_canonical_name():
    assert reg.resolve_particle_name("wet_diameter") == "Dwet"
    assert reg.resolve_particle_name("kappa") == "kappa"


def test_particle_builder_instantiation_provides_meta():
    builder = reg.get_particle_builder("Dwet")
    inst = builder({})
    assert inst.meta.name == "Dwet"
    axis_names = inst.meta.axis_names
    if isinstance(axis_names, str):
        axis_names = (axis_names,)
    assert axis_names == ("rh_grid",)


def test_unknown_particle_variable_suggests_close_match():
    with pytest.raises(reg.UnknownParticleVariableError) as excinfo:
        reg.resolve_particle_name("Dwetty")
    assert "Dwet" in str(excinfo.value)
