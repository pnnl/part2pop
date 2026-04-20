import numpy as np
import pytest

from PyParticle.population.builder import build_population
from PyParticle.analysis.population.factory import b_scat, b_abs

from tests.integration.adapters.pymiescatt_ref import HAS_PYMIESCATT, pymiescatt_adapter

pytestmark = pytest.mark.integration

@pytest.mark.skipif(not HAS_PYMIESCATT, reason="PyMieScatt not installed")
def test_homog_baseline_vs_pymiescatt():
    pop_cfg = {
        "type": "binned_lognormals",
        "N":   [1e9],
        "GMD": [100e-9],
        "GSD": [1.6],
        "N_bins": [60],
        "D_min": 1e-9,
        "D_max": 1e-6,
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
        "species_modifications": {
            "SO4": {"n_550": 1.45, "k_550": 0.0, "alpha_n": 0.0, "alpha_k": 0.0},
        },
        "D_is_wet": False,
    }
    var_cfg = {
        "morphology": "homogeneous",
        "rh_grid": [0.0],
        "wvl_grid": [450e-9, 550e-9, 650e-9],
        "T": 298.15,
        "species_modifications": pop_cfg["species_modifications"],
    }

    pop = build_population(pop_cfg)

    v_scat = b_scat.build(var_cfg)
    v_abs  = b_abs.build(var_cfg)
    arr_scat_pkg = np.asarray(v_scat.compute(pop), dtype=float).squeeze()
    arr_abs_pkg  = np.asarray(v_abs.compute(pop), dtype=float).squeeze()

    eps = 1e-18
    assert (arr_abs_pkg >= -eps).all(), f"Non-tiny negative absorption: min={arr_abs_pkg.min()}"
    arr_abs_pkg = np.where(arr_abs_pkg < 0, 0.0, arr_abs_pkg)
    arr_scat_pkg = np.where(arr_scat_pkg < 0, 0.0, arr_scat_pkg)

    wvl, bsc_ref, bab_ref = pymiescatt_adapter(pop_cfg, var_cfg)

    assert np.allclose(arr_scat_pkg, bsc_ref, rtol=1e-2, atol=1e-12), \
        f"b_scat mismatch\npkg={arr_scat_pkg}\nref={bsc_ref}"
    assert np.allclose(arr_abs_pkg, bab_ref, rtol=1e-2, atol=1e-12), \
        f"b_abs mismatch\npkg={arr_abs_pkg}\nref={bab_ref}"
