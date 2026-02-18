# tests/unit/population/factory/test_binned_lognormals.py

import numpy as np
import pytest

from part2pop import build_population


def _base_cfg():
    return {
        "type": "binned_lognormals",
        "GMD": [0.07e-6],
        "GSD": [1.6],
        "N": [1e8],
        "N_bins": [20],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
    }


def test_binned_lognormals_single_mode_total_number_and_bins():
    """
    Single lognormal mode binned into N_bins bins:
      - Population should have exactly N_bins particle IDs.
      - Total number concentration should be close to requested N.
    """
    cfg = {
        "type": "binned_lognormals",
        "GMD": [0.1e-6],
        "GSD": [1.6],
        "N": [1e8],
        "N_bins": [40],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
    }

    pop = build_population(cfg)

    # One mode, 40 bins
    assert len(pop.ids) == 40

    N_tot = float(pop.get_Ntot())
    # Discretization introduces small error; allow a bit of tolerance
    assert np.isclose(N_tot, 1e8, rtol=0.02)


def test_binned_lognormals_two_modes_multispecies():
    """
    Two lognormal modes with different sizes and a simple internal mixture:
      - Total number concentration ~ sum of mode N's.
      - Number of IDs is sum of per-mode N_bins.
    """
    N_modes = [5e7, 2e8]
    N_bins_modes = [20, 30]

    cfg = {
        "type": "binned_lognormals",
        "GMD": [0.05e-6, 0.15e-6],
        "GSD": [1.4, 1.7],
        "N": N_modes,
        "N_bins": N_bins_modes,
        # two species internally mixed in both modes
        "aero_spec_names": [["SO4", "OC"],["SO4", "OC"]],
        # shape: modes x species
        "aero_spec_fracs": [
            [0.7, 0.3],  # mode 1
            [0.4, 0.6],  # mode 2
        ],
    }

    pop = build_population(cfg)

    # Total number of bins = sum of N_bins across modes
    expected_ids = sum(N_bins_modes)
    assert len(pop.ids) == expected_ids

    # Total number concentration should be close to sum of N across modes
    N_requested = float(sum(N_modes))
    N_tot = float(pop.get_Ntot())
    assert np.isclose(N_tot, N_requested, rtol=0.02)
    
    # Basic check: number concentrations are non-negative
    assert np.all(pop.num_concs >= 0.0)

    # Basic check: mass concentrations are non-negative
    assert np.all(pop.spec_masses >= 0.0)


def test_requires_both_global_edges():
    cfg = _base_cfg()
    cfg["D_min"] = 0.01e-6
    with pytest.raises(ValueError, match="Provide both"):
        build_population(cfg)


def test_rejects_invalid_global_edges():
    cfg = _base_cfg()
    cfg["D_min"] = 0.3e-6
    cfg["D_max"] = 0.2e-6
    with pytest.raises(ValueError, match="D_min and D_max must be positive"):
        build_population(cfg)


def test_binned_lognormals_accepts_stringy_numbers():
    cfg = {
        "type": "binned_lognormals",
        "GMD": ["0.1e-6"],
        "GSD": ["1.6"],
        "N": ["1e8"],
        "N_bins": "40",
        "D_min": "0.01e-6",
        "D_max": "0.4e-6",
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
    }

    pop = build_population(cfg)
    assert len(pop.ids) == 40

    N_tot = float(pop.get_Ntot())
    assert np.isclose(N_tot, 1e8, rtol=0.02)
