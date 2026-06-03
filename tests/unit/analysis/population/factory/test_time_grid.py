import importlib
import numpy as np

import part2pop.analysis.population.factory.time_grid as time_grid_mod


def test_import_time_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.time_grid")


def test_time_grid_default_and_explicit_values():
    default_var = time_grid_mod.build({})
    default_out = default_var.compute(population=None)
    assert isinstance(default_out, np.ndarray)
    assert default_out.shape == (721,)
    assert np.allclose(default_out[:3], [0.0, 0.5, 1.0])
    assert np.isclose(default_out[-1], 360.0)

    var = time_grid_mod.build({"t_min": 1.0, "t_max": 2.0, "dt": 0.5})
    out = var.compute(population=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 1.5, 2.0])