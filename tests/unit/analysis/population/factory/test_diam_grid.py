import importlib
import numpy as np

import part2pop.analysis.population.factory.diam_grid as diam_grid_mod


def test_import_diam_grid():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.diam_grid")


def test_diam_grid_compute_array_and_dict():
    cfg = {"diam_grid": [1e-9, 2e-9, 5e-9]}
    var = diam_grid_mod.build(cfg)

    out = var.compute(population=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, [1e-9, 2e-9, 5e-9])

    out_dict = var.compute(population=None, as_dict=True)
    assert set(out_dict) == {"diam_grid"}
    assert isinstance(out_dict["diam_grid"], np.ndarray)
    assert np.allclose(out_dict["diam_grid"], out)
