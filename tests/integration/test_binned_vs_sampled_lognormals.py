import numpy as np
import pytest

from part2pop.population import build_population
from part2pop.analysis.population.factory.b_scat import BScatVar
from part2pop.analysis.population.factory.dNdlnD import DNdlnDVar


def _lognormal_cfg_common():
    """Common physical configuration: a 2-mode sulfate-only lognormal distribution."""
    N_modes = [5.0e8, 1.0e8]       # [#/m^3]
    GMD_modes = [100e-9, 30e-9]    # [m]
    GSD_modes = [1.6, 1.6]

    spec_names_per_mode = [["SO4"], ["SO4"]]
    spec_fracs_per_mode = [[1.0], [1.0]]

    return {
        "N": N_modes,
        "GMD": GMD_modes,
        "GSD": GSD_modes,
        "aero_spec_names": spec_names_per_mode,
        "aero_spec_fracs": spec_fracs_per_mode,
    }


def _build_binned_population(n_bins=200):
    common = _lognormal_cfg_common()
    cfg = {
        "type": "binned_lognormals",
        "N_bins": int(n_bins),
        "N_sigmas": 6.0,
        **common,
    }
    return build_population(cfg)


def _build_sampled_population(n_part=20000, seed=42):
    """
    Sampled lognormal population meant to represent the *same* physical
    distribution as _build_binned_population.
    """
    common = _lognormal_cfg_common()
    cfg = {
        "type": "sampled_lognormals",
        "n_part": int(n_part),
        "N_sigmas": 6.0,
        "seed": seed,
        **common,
    }
    return build_population(cfg)


def _get_totals(pop):
    """
    Extract total number and total dry mass using the standard population API.
    Adjust names only if your actual methods differ.
    """
    if hasattr(pop, "get_total_number"):
        Ntot = pop.get_total_number()
    else:
        Ntot = pop.get_Ntot()

    if hasattr(pop, "get_total_dry_mass"):
        mtot = pop.get_total_dry_mass()
    else:
        mtot = pop.get_tot_dry_mass()

    return float(Ntot), float(mtot)


def _get_dNdlnD(pop, n_bins=80):
    """
    Compute dN/dlnD on a common grid using the existing DNdlnDVar API.

    IMPORTANT: DNdlnDVar.compute(pop) in your code does **not** return a
    tuple. It returns:
        - a 1D array (dens) if as_dict=False, or
        - a dict with keys {"D", "dNdlnD", "edges"} if as_dict=True.

    To get both the diameter grid and the values, we must use as_dict=True
    and then convert D -> lnD for comparison.
    """
    var = DNdlnDVar({"N_bins": n_bins})

    # This uses your standard analysis-variable pattern.
    out = var.compute(pop, as_dict=True)

    # Expect your existing keys; if they differ, fix here, not in analysis code.
    D = np.asarray(out["D"], dtype=float)
    dNdlnD = np.asarray(out["dNdlnD"], dtype=float)

    # Convert to lnD for comparison
    lnD = np.log(D)

    return lnD, dNdlnD


def _get_b_scat(pop):
    """
    Compute b_scat at 550 nm, RH = 0 using your BScatVar wrapper.
    """
    cfg = {
        "rh_grid": [0.0],
        "wvl_grid": [550e-9],
        "morphology": "core-shell",
        "T": 298.15,
    }
    var = BScatVar(cfg)
    arr = var.compute(pop, as_dict=False)
    # arr should be [nrh, nwvl]; here (1,1). Squeeze to scalar.
    return float(np.squeeze(arr))


@pytest.mark.parametrize("n_bins, n_part", [(200, 50000)])
def test_binned_vs_sampled_lognormals_consistency(n_bins, n_part):
    """
    Integration test:

    For a given physical lognormal configuration, compare:

      - 'binned_lognormals' population
      - 'sampled_lognormals' population

    They should agree (within sampling noise) on:
      - total number and total dry mass
      - dN/dlnD on a common grid
      - b_scat at 550 nm, RH = 0

    This uses the existing analysis variable APIs and does **not** modify
    any analysis code.
    """
    pop_binned = _build_binned_population(n_bins=n_bins)
    pop_sampled = _build_sampled_population(n_part=n_part, seed=1234)

    # --- Totals: N and mass should be close --------------------------------
    N_bin, m_bin = _get_totals(pop_binned)
    N_smp, m_smp = _get_totals(pop_sampled)

    # Allow a couple percent for Monte Carlo noise
    assert np.isclose(N_bin, N_smp, rtol=0.02), (
        f"Total number mismatch: binned={N_bin}, sampled={N_smp}"
    )
    assert np.isclose(m_bin, m_smp, rtol=0.02), (
        f"Total mass mismatch: binned={m_bin}, sampled={m_smp}"
    )

    # --- dN/dlnD: compare on common grid via DNdlnDVar ---------------------
    lnD_bin, dNdlnD_bin = _get_dNdlnD(pop_binned, n_bins=80)
    lnD_smp, dNdlnD_smp = _get_dNdlnD(pop_sampled, n_bins=80)

    # Grids should match because we used the same N_bins
    assert lnD_bin.shape == lnD_smp.shape
    assert np.allclose(lnD_bin, lnD_smp, rtol=1e-12, atol=1e-12)

    # Allow sampling noise but not a systematic bias.
    max_abs = np.max(np.abs(dNdlnD_bin))
    if max_abs == 0.0:
        rel_diff = 0.0
    else:
        rel_diff = np.max(np.abs(dNdlnD_bin - dNdlnD_smp)) / max_abs

    assert rel_diff < 0.05, (
        f"dN/dlnD mismatch between binned and sampled lognormals; "
        f"max_rel_diff={rel_diff}"
    )

    # --- b_scat: this is the optical consistency check ---------------------
    b_bin = _get_b_scat(pop_binned)
    b_smp = _get_b_scat(pop_sampled)

    # If weights and optics are consistent, these should be within sampling noise.
    assert np.isclose(b_bin, b_smp, rtol=0.05), (
        f"b_scat mismatch: binned={b_bin}, sampled={b_smp}"
    )
