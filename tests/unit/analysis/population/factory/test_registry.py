import pytest

from part2pop.analysis.population.factory import registry
from part2pop.analysis.population.factory.wvl_grid import WvlGridVar


def test_list_variables_discovers_factories():
    names = registry.list_variables()
    # Core variables should be discoverable by name
    assert "b_abs" in names
    assert "wvl_grid" in names


def test_alias_resolution_and_builder():
    builder = registry.get_population_builder("wvls")
    inst = builder({"wvl_grid": [550e-9]})
    assert isinstance(inst, WvlGridVar)


def test_unknown_variable_raises():
    with pytest.raises(registry.UnknownVariableError):
        registry.get_population_builder("not_real_var")
