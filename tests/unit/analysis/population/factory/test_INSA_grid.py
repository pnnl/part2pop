import importlib
import numpy as np

import part2pop.analysis.population.factory.INSA_grid as insa_grid_mod
from part2pop.analysis.population.factory import registry as pop_factory_registry


def test_import_insa_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.INSA_grid")


def test_insa_grid_compute_array_and_dict():
    var = insa_grid_mod.build({"insa_grid": [1e-12, 2e-12, 4e-12]})

    out = var.compute(population=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, [1e-12, 2e-12, 4e-12])

    out_dict = var.compute(population=None, as_dict=True)
    assert set(out_dict) == {"INSA_grid"}
    assert isinstance(out_dict["INSA_grid"], np.ndarray)
    assert np.allclose(out_dict["INSA_grid"], out)


def test_insa_grid_registry_alias_and_value_key_consistency():
    assert pop_factory_registry.resolve_name("insa_grid") == "INSA_grid"
    builder = pop_factory_registry.get_population_builder("insa_grid")
    var = builder({"insa_grid": [1e-12]})
    assert var.meta.name == "INSA_grid"
    assert "insa_grid" in var.meta.aliases