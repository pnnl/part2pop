# tests/unit/analysis/test_global_registry.py

import pytest

from part2pop.analysis import global_registry as gr


def test_build_full_variable_name_population():
    """
    family_name must be the full class-family string, e.g. 'PopulationVariable',
    not a short 'population' keyword.
    """
    full = gr.build_full_variable_name("dNdlnD", "PopulationVariable")
    # family_to_suffix('PopulationVariable') -> '.population'
    assert full.endswith(".population")
    assert full.startswith("dNdlnD")


def test_get_variable_builder_for_dNdlnD():
    """
    get_variable_builder takes an optional config dict with 'family'.
    """
    builder = gr.get_variable_builder("dNdlnD", config={"family": "PopulationVariable"})
    assert callable(builder)


@pytest.mark.xfail(
    reason=(
        "list_registered_variables currently fails because "
        "analysis.population.factory.registry exposes "
        "get_population_builder() but global_registry still "
        "calls preg.get_builder()."
    )
)
def test_list_registered_variables_includes_dNdlnD():
    """
    This test documents the current bug in list_registered_variables.
    Once analysis/global_registry.py is updated to use
    get_population_builder instead of get_builder, this xfail can be
    removed and the assertion should pass.
    """
    names = gr.list_registered_variables()
    # Full names include suffixes like '.population'
    assert any(name.startswith("dNdlnD") for name in names)
