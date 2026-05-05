import importlib
import numpy as np

import part2pop.analysis.population.factory.INSA_grid as insa_grid_mod


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
    assert set(out_dict) == {"insa_grid"}
    assert isinstance(out_dict["insa_grid"], np.ndarray)
    assert np.allclose(out_dict["insa_grid"], out)