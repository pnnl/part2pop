import importlib
import numpy as np

import part2pop.analysis.population.factory.s_grid as s_grid_mod

def test_import_s_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.s_grid")


def test_s_grid_explicit_and_alias_with_dict_output():
    var = s_grid_mod.build({"s_grid": [0.1, 0.2, 0.5]})
    out = var.compute(population=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, [0.1, 0.2, 0.5])

    out_dict = var.compute(population=None, as_dict=True)
    assert set(out_dict) == {"s_grid"}
    assert np.allclose(out_dict["s_grid"], out)

    alias_var = s_grid_mod.build({"s_eval": [0.05, 0.15]})
    alias_out = alias_var.compute(population=None)
    assert isinstance(alias_out, np.ndarray)
    assert alias_out.shape == (2,)
    assert np.allclose(alias_out, [0.05, 0.15])
