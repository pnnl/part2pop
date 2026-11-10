from copy import deepcopy

import numpy as np
import pytest

from part2pop.population.factory import binned_lognormals


def config(global_edges=True):
    cfg = {"N": [1e8], "GMD": [1e-7], "GSD": [1.2], "N_bins": 3,
           "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.]]}
    if global_edges:
        cfg.update(D_min=1e-8, D_max=1e-6)
    return cfg


@pytest.mark.parametrize("global_edges", [False, True])
@pytest.mark.parametrize("key,value", [
    ("N", -1.), ("N", np.nan), ("N", np.inf),
    ("GMD", 0.), ("GMD", -1.), ("GMD", np.nan), ("GMD", np.inf),
    ("GSD", 1.), ("GSD", .9), ("GSD", np.nan), ("GSD", np.inf),
])
def test_invalid_mode_parameters_rejected(key, value, global_edges):
    cfg = config(global_edges)
    cfg[key] = [value]
    with pytest.raises(ValueError, match=key):
        binned_lognormals.build(cfg)


@pytest.mark.parametrize("key,value", [("D_max", np.inf), ("D_min", np.nan)])
def test_nonfinite_global_edges_rejected(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError, match="D_min|D_max"):
        binned_lognormals.build(cfg)


@pytest.mark.parametrize("value", [0., -1., np.inf, np.nan])
def test_invalid_automatic_grid_width_rejected(value):
    cfg = config(False)
    cfg["N_sigmas"] = value
    with pytest.raises(ValueError, match="N_sigmas"):
        binned_lognormals.build(cfg)


def test_grid_without_sampled_support_raises_instead_of_nan():
    cfg = config()
    cfg["GMD"] = [1e-3]
    with np.errstate(invalid="raise", divide="raise"):
        with pytest.raises(ValueError, match="support"):
            binned_lognormals.build(cfg)


@pytest.mark.parametrize("global_edges", [False, True])
@pytest.mark.parametrize("number", [0., 1e8, 1e308])
def test_valid_modes_keep_full_number_and_leave_config_unchanged(global_edges, number):
    cfg = config(global_edges)
    cfg["N"] = [number]
    before = deepcopy(cfg)
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        pop = binned_lognormals.build(cfg)
    assert cfg == before
    assert np.all(np.isfinite(pop.num_concs))
    assert np.all(pop.num_concs >= 0)
    assert pop.get_Ntot() == pytest.approx(number)


def test_zero_number_mode_keeps_zero_counts_even_without_sampled_support():
    cfg = config()
    cfg.update(N=[0.], GMD=[1e-3])
    with np.errstate(invalid="raise", divide="raise"):
        pop = binned_lognormals.build(cfg)
    assert len(pop.ids) == 3
    np.testing.assert_array_equal(pop.num_concs, np.zeros(3))
