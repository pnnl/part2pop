import importlib
import numpy as np

import part2pop.analysis.population.factory.wvl_grid as wvl_grid_mod

def test_import_wvl_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.wvl_grid")


def test_wvl_grid_default_explicit_and_dict_output():
    default_var = wvl_grid_mod.build({})
    default_out = default_var.compute(population=None)
    assert isinstance(default_out, np.ndarray)
    assert default_out.shape == (0,)

    var = wvl_grid_mod.build({"wvl_grid": [500e-9, 700e-9]})
    out = var.compute(population=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
    assert np.allclose(out, [500e-9, 700e-9])

    out_dict = var.compute(population=None, as_dict=True)
    assert set(out_dict) == {"wvl_grid"}
    assert np.allclose(out_dict["wvl_grid"], out)
