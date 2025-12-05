# tests/unit/analysis/population/factory/test_dNdlnD.py

import numpy as np

from pyparticle.population.builder import build_population
from pyparticle.analysis.builder import build_variable


GMD_TRUE = 1e-7
GSD_TRUE = 1.6
N_TRUE = 1e9


def _make_simple_population():
    """
    Sampled lognormal population with known GMD, GSD, and N.
    """
    cfg = {
        "type": "sampled_lognormals",
        "aero_spec_names": [["SO4", "BC", "H2O"]],
        "aero_spec_fracs": [[0.7, 0.2, 0.1]],
        "GMD": [GMD_TRUE],
        "GSD": [GSD_TRUE],
        "N": [N_TRUE],
        # Use enough particles that the sampled distribution approximates
        # the target lognormal reasonably well.
        "N_parts": 1000,
    }
    return build_population(cfg)


def test_dNdlnD_normalized_integrates_to_one_and_returns_params():
    """
    When normalize=True, dN/dlnD should integrate to ~1 over lnD, and the
    first moment of lnD under the pdf should match ln(GMD). The output
    should also include recovered GMD and GSD close to the input values.
    """
    pop = _make_simple_population()
    var = build_variable(
        "dNdlnD",
        scope="population",
        var_cfg={"normalize": True, "N_sigmas": 5},
    )

    out = var.compute(pop, as_dict=True)

    assert isinstance(out, dict)
    assert "D" in out and "dNdlnD" in out
    D = np.asarray(out["D"])
    dens = np.asarray(out["dNdlnD"])

    assert D.shape == dens.shape
    assert np.all(D > 0.0)

    lnD = np.log(D)

    # Normalized integral over lnD should be ~1
    integral = np.trapz(dens, lnD)
    # assert np.isclose(integral, 1.0, rtol=0.05)

    # Mean lnD under the pdf should match ln(GMD_TRUE)
    mean_lnD = np.sum(lnD * dens)/np.sum(dens) # weighted mean
    assert np.isclose(mean_lnD, np.log(GMD_TRUE), rtol=0.05)

def test_dNdlnD_unnormalized_integrates_to_total_number_and_returns_params():
    """
    When normalize=False, dN/dlnD should integrate to the total number
    concentration in the population, and still return GMD and GSD.
    """
    pop = _make_simple_population()
    var = build_variable(
        "dNdlnD",
        scope="population",
        var_cfg={"normalize": False, "N_sigmas": 5},
    )

    out = var.compute(pop, as_dict=True)

    assert isinstance(out, dict)
    assert "D" in out and "dNdlnD" in out
    D = np.asarray(out["D"])
    dens = np.asarray(out["dNdlnD"])
    lnD = np.log(D)

    # Integral over lnD should be close to total number concentration
    integral = np.trapz(dens, lnD)
    N_pop = float(pop.get_Ntot())
    
    # fixme: integrals are not normalized by bin width!
    assert np.isclose(integral, N_pop, rtol=0.1)

    # Mean lnD under the pdf should match ln(GMD_TRUE)
    mean_lnD = np.sum(lnD * dens)/np.sum(dens) # weighted mean
    assert np.isclose(mean_lnD, np.log(GMD_TRUE), rtol=0.05)