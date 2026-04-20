import numpy as np
import pytest

from PyParticle.population.builder import build_population
from PyParticle.analysis.population.factory import b_scat, b_abs

from tests.integration.adapters.pymiescatt_ref import HAS_PYMIESCATT, pymiescatt_adapter_core_shell

pytestmark = pytest.mark.integration

@pytest.mark.skipif(not HAS_PYMIESCATT, reason="PyMieScatt not installed")
def test_core_shell_baseline_vs_pymiescatt():
    pop_cfg = {
        "type": "binned_lognormals",
        "N":   [1e9],
        "GMD": [150e-9],
        "GSD": [1.6],
        "N_bins": [80],
        "D_min": 50e-9, "D_max": 500e-9,
        "aero_spec_names": [["BC","SO4"]],
        "aero_spec_fracs": [[0.2, 0.8]],  # mass fractions
        "species_modifications": {
            "BC":  {"n_550": 1.85, "k_550": 0.80, "alpha_n": 0.0, "alpha_k": 0.0},
            "SO4": {"n_550": 1.45, "k_550": 0.00, "alpha_n": 0.0, "alpha_k": 0.0},
        },
        "D_is_wet": False,
    }
    var_cfg = {
        "morphology": "core-shell",
        "rh_grid": [0.0],
        "wvl_grid": [450e-9, 550e-9, 650e-9],
        "T": 298.15,
        "species_modifications": pop_cfg["species_modifications"],
    }

    pop = build_population(pop_cfg)

    arr_scat_pkg = np.asarray(b_scat.build(var_cfg).compute(pop), float).squeeze()
    arr_abs_pkg  = np.asarray(b_abs.build(var_cfg).compute(pop),  float).squeeze()

    eps = 1e-18
    assert (arr_abs_pkg >= -eps).all()
    arr_abs_pkg  = np.where(arr_abs_pkg < 0, 0.0, arr_abs_pkg)
    arr_scat_pkg = np.where(arr_scat_pkg < 0, 0.0, arr_scat_pkg)

    wvl, bsc_ref, bab_ref = pymiescatt_adapter_core_shell(pop_cfg, var_cfg, core_specs=("BC",))

    assert np.allclose(arr_scat_pkg, bsc_ref, rtol=2e-2, atol=1e-12), \
        f"b_scat mismatch\npkg={arr_scat_pkg}\nref={bsc_ref}"
    assert np.allclose(arr_abs_pkg,  bab_ref, rtol=2e-2, atol=1e-12), \
        f"b_abs mismatch\npkg={arr_abs_pkg}\nref={bab_ref}"
