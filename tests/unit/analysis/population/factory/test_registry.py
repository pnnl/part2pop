import pytest

from part2pop.analysis.population.factory import registry as reg


def test_population_registry_discovers_core_variables():
    names = reg.list_variables(include_aliases=True)
    assert "dNdlnD" in names
    assert "T_grid" in names
    assert "temp" in names


def test_resolve_alias_and_canonical_names():
    assert reg.resolve_name("temp") == "T_grid"
    assert reg.resolve_name("dNdlnD") == "dNdlnD"


def test_population_builder_instantiation_provides_meta():
    builder = reg.get_population_builder("dNdlnD")
    inst = builder({})
    assert inst.meta.name == "dNdlnD"
    assert "D" in inst.meta.axis_names


def test_unknown_variable_suggests_existing_name():
    with pytest.raises(reg.UnknownVariableError) as excinfo:
        reg.resolve_name("dNd")
    assert "dNdlnD" in str(excinfo.value)
