# tests/integration/test_inp_model.py

"""
Integration test: INP freezing efficiency for monodisperse vs lognormal populations.
Inspired by Alpert & Knopf (2016) synthetic comparison of unfrozen fractions.
"""

import numpy as np
import pytest

from part2pop.population import build_population
from part2pop.freezing import build_freezing_population

# Freezing kinetics from Alpert & Knopf (2016)
AK_TIME = np.array([
    0.0, 14.99, 29.98, 44.98, 59.98, 74.96, 89.98, 104.97, 119.98, 134.97,
    149.97, 164.98, 179.98, 194.99, 209.98, 224.99, 239.98, 254.98, 269.98,
    284.98, 299.98, 314.98, 329.98, 344.98, 359.97
])

AK_MONO_BOUNDS = np.array([
    [1.0, 0.8564, 0.7276, 0.6210, 0.5254, 0.4452, 0.3821, 0.3286, 0.2801, 0.2366,
     0.2033, 0.1731, 0.1477, 0.1254, 0.1068, 0.0913, 0.0766, 0.0650, 0.0547,
     0.0464, 0.0389, 0.0328, 0.0272, 0.0230, 0.0195],
    [1.0, 0.8987, 0.7753, 0.6715, 0.5791, 0.5053, 0.4450, 0.3795, 0.3301, 0.2857,
     0.2469, 0.2151, 0.1855, 0.1626, 0.1405, 0.1232, 0.1072, 0.0932, 0.0810,
     0.0709, 0.0622, 0.0545, 0.0476, 0.0416, 0.0370]
])

AK_LOGN_BOUNDS = np.array([
    [1.0, 0.6873, 0.5849, 0.5229, 0.4725, 0.4417, 0.4128, 0.3896, 0.3696, 0.3518,
     0.3365, 0.3240, 0.3113, 0.3013, 0.2906, 0.2826, 0.2738, 0.2648, 0.2585,
     0.2514, 0.2461, 0.2402, 0.2328, 0.2280, 0.2232],
    [1.0, 0.7394, 0.6326, 0.5675, 0.5210, 0.4834, 0.4578, 0.4337, 0.4129, 0.3937,
     0.3793, 0.3653, 0.3508, 0.3406, 0.3295, 0.3198, 0.3107, 0.3031, 0.2954,
     0.2872, 0.2798, 0.2752, 0.2699, 0.2635, 0.2585]
])

@pytest.mark.integration
def test_inp_freezing_comparison_to_alpert_knopf():
    """Compare simulated freezing behavior against bounds from Alpert & Knopf (2016)."""

    # Reconstruct the INSA distribution and population configs
    np.random.seed(0)
    INSA = np.random.lognormal(mean=np.log(1e-9), sigma=np.log(10.0), size=3000)
    Dps = list(2.0 * np.sqrt(INSA / (4 * np.pi)))
    Ns = [1e9 / len(Dps)] * len(Dps)
    common_mod = {'SO4': {'m_log10Jhet': 0.0, 'b_log10Jhet': 7.0}}

    lognormal_cfg = {
        "type": "monodisperse",
        "N": Ns,
        "D": Dps,
        "aero_spec_names": [["SO4"]] * len(Dps),
        "aero_spec_fracs": [[1.0]] * len(Dps),
        "species_modifications": common_mod
    }

    monodisperse_cfg = {
        "type": "monodisperse",
        "N": [1e9],
        "D": [1.78e-5],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
        "species_modifications": common_mod
    }

    # Build freezing populations
    var_cfg = {
        "morphology": "homogeneous",
        "T_grid": [-30],
        "T_units": "C",
        "species_modifications": common_mod
    }

    mono = build_freezing_population(build_population(monodisperse_cfg), var_cfg)
    logn = build_freezing_population(build_population(lognormal_cfg), var_cfg)

    # Simulate freezing over time
    time = AK_TIME
    mono_Nfrz = np.zeros((len(time), mono.num_concs.shape[0]))
    logn_Nfrz = np.zeros((len(time), logn.num_concs.shape[0]))

    for i, dt in enumerate(time):
        mono_Nfrz[i] = mono.num_concs * (1 - np.exp(-mono.Jhet * mono.INSA * dt))
        logn_Nfrz[i] = logn.num_concs * (1 - np.exp(-logn.Jhet * logn.INSA * dt))

    mono_unfrz = (np.sum(mono.num_concs) - np.sum(mono_Nfrz, axis=1)) / np.sum(mono.num_concs)
    logn_unfrz = (np.sum(logn.num_concs) - np.sum(logn_Nfrz, axis=1)) / np.sum(logn.num_concs)

    # Assert simulation curves are within Alpert & Knopf experimental bounds
    assert np.all(mono_unfrz <= AK_MONO_BOUNDS[1] + 1e-4)
    assert np.all(mono_unfrz >= AK_MONO_BOUNDS[0] - 1e-4)
    assert np.all(logn_unfrz <= AK_LOGN_BOUNDS[1] + 1e-4)
    assert np.all(logn_unfrz >= AK_LOGN_BOUNDS[0] - 1e-4)
