import numpy as np

from part2pop.analysis.population.factory.wvl_grid import WvlGridVar, build


def test_build_returns_wavelength_grid():
    var = build({"wvl_grid": [400e-9]})
    assert isinstance(var, WvlGridVar)
    assert var.meta.aliases == ("wvls",)


def test_compute_handles_missing_grid():
    var = build({})
    out = var.compute(as_dict=True)
    assert out["wvl_grid"].size == 0


def test_compute_returns_numpy_array():
    cfg = {"wvl_grid": [400e-9, 550e-9]}
    var = build(cfg)
    arr = var.compute()
    assert np.allclose(arr, np.array(cfg["wvl_grid"]))
