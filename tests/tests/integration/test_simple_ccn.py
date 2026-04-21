# tests/integration/test_ccn.py
# Integration test comparing CCN fraction and number concentration consistency

import numpy as np
import pytest
from part2pop import build_population
from part2pop.analysis.population.factory.Nccn import build as build_nccn
from part2pop.analysis.population.factory.frac_ccn import build as build_frac_ccn

@pytest.mark.integration
def test_ccn_fraction_and_number_consistency():
    """Test CCN fraction and number concentration agree with expected activation behavior."""

    # Define three monodisperse populations with known diameters and number concentrations
    D_vals = [50e-9, 100e-9, 200e-9]
    N_vals = [1e5, 2e5, 3e5]  # number concentrations in m^-3
    pop_cfg = {
        "type": "monodisperse",
        "D": D_vals,
        "N": N_vals,
        "aero_spec_names": [["SO4"]] * 3,
        "aero_spec_fracs": [[1.0]] * 3,
    }
    pop = build_population(pop_cfg)

    # Supersaturation levels (in percent)
    s_grid = [0.0, 0.3, 1.0]
    T = 298.15  # Kelvin
    ccn_cfg = {"s_grid": s_grid, "T": T}

    # Build and compute both CCN number and fraction
    nccn_result = build_nccn(ccn_cfg).compute(pop, as_dict=True)
    frac_result = build_frac_ccn(ccn_cfg).compute(pop, as_dict=True)

    Nccn_values = np.array(nccn_result["Nccn"])
    frac_values = np.array(frac_result["frac_ccn"])
    total_N = sum(N_vals)

    # Consistency: Nccn ≈ frac * total_N
    np.testing.assert_allclose(Nccn_values, frac_values * total_N, rtol=1e-4)

    # Specific expectations
    assert Nccn_values[0] == 0
    assert frac_values[0] == 0

    expected_mid = N_vals[1] + N_vals[2]  # 100nm + 200nm activated
    np.testing.assert_allclose(Nccn_values[1], expected_mid, rtol=0.02)

    np.testing.assert_allclose(Nccn_values[2], total_N, rtol=1e-8)
    np.testing.assert_allclose(frac_values[2], 1.0, rtol=1e-8)
