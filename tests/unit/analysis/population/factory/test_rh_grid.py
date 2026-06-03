import importlib
import numpy as np

import part2pop.analysis.population.factory.rh_grid as rh_grid_mod

def test_import_rh_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.rh_grid")


def test_rh_grid_default_and_explicit_with_dict_output():
    default_var = rh_grid_mod.build({})
    default_out = default_var.compute(population=None)
    assert isinstance(default_out, np.ndarray)
    assert default_out.shape == (1,)
    assert np.allclose(default_out, [0.0])

    var = rh_grid_mod.build({"rh_grid": [0.25, 0.5, 0.9]})
    out = var.compute(population=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, [0.25, 0.5, 0.9])

    out_dict = var.compute(population=None, as_dict=True)
    assert set(out_dict) == {"rh_grid"}
    assert isinstance(out_dict["rh_grid"], np.ndarray)
    assert np.allclose(out_dict["rh_grid"], out)
