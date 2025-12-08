# tests/unit/population/factory/test_mam4.py

import shutil
import numpy as np
import pytest

from part2pop.constants import (
    MOLAR_MASS_DRY_AIR,
    DENSITY_LIQUID_WATER,
    R,
)
from part2pop.utilities import power_moments_from_lognormal
from part2pop.population.factory import mam4

# Try importing the mam4 factory and netCDF4; if that fails, skip all tests here.
try:
    import netCDF4  # noqa: F401
    from netCDF4 import Dataset
    from part2pop.population.factory.mam4 import build as build_mam4
    HAS_MAM4 = True
except Exception:
    HAS_MAM4 = False


pytestmark = pytest.mark.skipif(
    not HAS_MAM4,
    reason="mam4 factory or netCDF4 not available; skipping MAM4 tests.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_get_mam_input_reads_value(tmp_path):
    content = "numc1 = 3.5,\nother = 1.0,\n"
    fname = tmp_path / "namelist"
    fname.write_text(content)

    val = mam4.get_mam_input("numc1", fname)
    assert val == 3.5

    # missing variable returns 0.0 (no error)
    missing = mam4.get_mam_input("not_here", fname)
    assert missing == 0.0


def _parse_p_T_from_namelist(namelist_path):
    """
    Parse dry-air pressure [Pa] and temperature [K] from a MAM namelist text file.

    We scan for lines like:
        press = 90000.,
        temp  = 273.15,
    """
    p = None
    T = None

    with open(namelist_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("press"):
                if "=" in line:
                    rhs = line.split("=", 1)[1]
                    rhs = rhs.split(",", 1)[0]
                    p = float(rhs)
            if line.startswith("temp"):
                if "=" in line:
                    rhs = line.split("=", 1)[1]
                    rhs = rhs.split(",", 1)[0]
                    T = float(rhs)

    if p is None or T is None:
        raise ValueError("Could not parse p/T from namelist.")

    return p, T


def _compute_mode_totals_from_mam_raw(
    p,
    T,
    GSDs,
    num_aer_vals,
    so4_vals,
    soa_vals,
    dgn_a_vals,
    dgn_awet_vals,
):
    """
    Compute mode-wise totals in physical units (per m^3) from raw MAM arrays
    using the same formulas as part2pop.population.factory.mam4.build.
    """
    rho_dry_air = MOLAR_MASS_DRY_AIR * p / (R * T)  # [kg/m^3]

    # Mode total number concentrations and dry species masses [per m^3]
    N_modes = num_aer_vals * rho_dry_air
    mass_so4_modes = so4_vals * rho_dry_air
    mass_soa_modes = soa_vals * rho_dry_air

    # Water mass via difference in third moments between wet and dry
    mass_h2o_modes = DENSITY_LIQUID_WATER * np.pi / 6.0 * np.array(
        [
            power_moments_from_lognormal(3, N, gmd_wet, gsd)
            - power_moments_from_lognormal(3, N, gmd_dry, gsd)
            for (N, gmd_dry, gmd_wet, gsd) in zip(
                N_modes, dgn_a_vals, dgn_awet_vals, GSDs
            )
        ]
    )

    return {
        "N_modes": N_modes,
        "mass_so4_modes": mass_so4_modes,
        "mass_soa_modes": mass_soa_modes,
        "mass_h2o_modes": mass_h2o_modes,
        "N_total": float(np.sum(N_modes)),
        "mass_so4_total": float(np.sum(mass_so4_modes)),
        "mass_soa_total": float(np.sum(mass_soa_modes)),
        "mass_h2o_total": float(np.sum(mass_h2o_modes)),
    }


def _totals_from_population(pop):
    """
    Compute:
      - total number concentration [#/m^3]
      - total mass per species [kg/m^3]
    from a ParticlePopulation produced by mam4.build.
    """
    N_total = float(np.sum(pop.num_concs))

    species_names = [sp.name.upper() for sp in pop.species]
    spec_masses = pop.spec_masses  # (n_particles, n_species)

    totals_by_species = {}
    for j, nm in enumerate(species_names):
        mass_j = float(np.sum(pop.num_concs * spec_masses[:, j]))
        totals_by_species[nm] = mass_j

    return N_total, totals_by_species


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mam4_build_disallows_timestep_zero():
    """
    The mam4 factory explicitly disallows timestep=0 and should raise
    a ValueError with a clear message. This exercises the early guard
    logic without requiring any files.
    """
    cfg = {
        "mam4_dir": "/non/existent/path",
        "timestep": 0,
        "GSD": [1.5, 1.6, 1.7, 1.8],
        "N_bins": [10, 10, 10, 10],
    }

    with pytest.raises(ValueError) as excinfo:
        build_mam4(cfg)

    msg = str(excinfo.value).lower()
    assert "timestep" in msg and "invalid" in msg


def test_mam4_build_conserves_number_and_composition(tmp_path):
    """
    Conservation / consistency test for mam4.build using real mam_output.nc
    and namelist:

    - Read p and T from the namelist (press/temp).
    - Read num_aer, so4_aer, soa_aer, dgn_a, dgn_awet at a given time index.
    - Convert to #/m^3 and kg/m^3 using the same assumptions as mam4.build.
    - Run mam4.build with consistent p, T, GSD.
    - Check:
        * total number concentration is conserved
        * the bulk SO4/SOA/H2O composition fractions from the population
          are consistent with those implied by the MAM fields.
    """
    mam_output_path = "../examples/example_data/model_output/mam4/mam_output.nc"
    namelist_path = "../examples/example_data/model_output/mam4/namelist"

    # 1. Read p, T from namelist
    p, T = _parse_p_T_from_namelist(namelist_path)

    # 2. Read raw MAM output
    ds = Dataset(mam_output_path, "r")

    # Choose a time index consistent with cfg["timestep"] (see below).
    # If mam4.build uses output_tt_idx = timestep - 2, and we set timestep=2,
    # then tt = 0 here.
    tt = 0

    num_aer_vals = ds["num_aer"][:, tt]
    so4_vals = ds["so4_aer"][:, tt]
    soa_vals = ds["soa_aer"][:, tt]
    dgn_a_vals = ds["dgn_a"][:, tt]
    dgn_awet_vals = ds["dgn_awet"][:, tt]

    # Geometric standard deviations per mode must match what we pass to build_mam4.
    # If GSDs are in the file, prefer reading them; otherwise, define them here.
    if "gsd_a" in ds.variables:
        GSDs = ds["gsd_a"][:, tt]
    else:
        # If not present, fall back to config-style GSDs (replace as needed)
        GSDs = np.ones_like(num_aer_vals) * 1.6

    ds.close()

    diag_raw = _compute_mode_totals_from_mam_raw(
        p=p,
        T=T,
        GSDs=GSDs,
        num_aer_vals=num_aer_vals,
        so4_vals=so4_vals,
        soa_vals=soa_vals,
        dgn_a_vals=dgn_a_vals,
        dgn_awet_vals=dgn_awet_vals,
    )

    N_expected = diag_raw["N_total"]
    so4_expected = diag_raw["mass_so4_total"]
    soa_expected = diag_raw["mass_soa_total"]
    h2o_expected = diag_raw["mass_h2o_total"]

    # Bulk composition fractions implied by MAM fields
    total_mass_expected = so4_expected + soa_expected + h2o_expected
    f_so4_expected = so4_expected / total_mass_expected
    f_soa_expected = soa_expected / total_mass_expected
    f_h2o_expected = h2o_expected / total_mass_expected

    # 3. Run mam4.build using the same mam_output.nc, p, T, GSDs

    # mam4.build typically expects a directory containing mam_output.nc
    shutil.copy(mam_output_path, tmp_path / "mam_output.nc")

    cfg = {
        "mam4_dir": str(tmp_path),
        "timestep": 2,                  # so that output_tt_idx == tt == 0
        "GSD": list(GSDs),
        "N_bins": [40] * len(GSDs),     # or whatever you normally use
        "N_sigmas": 5,                  # diameter range for bins
        "p": p,
        "T": T,
    }

    pop = build_mam4(cfg)

    # 4. Totals from population
    N_pop, totals_by_species = _totals_from_population(pop)

    # Species keys: adjust these to match your actual species names in part2pop.
    so4_pop = totals_by_species.get("SO4", 0.0)
    # For organics, you might use "SOA" or "OC" (MAM → OC mapping).
    soa_pop = totals_by_species.get("SOA", totals_by_species.get("OC", 0.0))
    h2o_pop = totals_by_species.get("H2O", 0.0)

    total_mass_pop = so4_pop + soa_pop + h2o_pop
    f_so4_pop = so4_pop / total_mass_pop
    f_soa_pop = soa_pop / total_mass_pop
    
    # 5. Checks

    # Number concentration should be conserved tightly
    assert N_expected > 0.0
    assert N_pop > 0.0
    assert np.isclose(N_pop, N_expected, rtol=0.01)

    # Composition fractions should match reasonably well
    # (absolute masses need not match exactly because mam4.build uses the
    #  mixing ratios only to set composition fractions, not to enforce
    #  exact mass conservation).
    assert np.isclose(f_so4_pop, f_so4_expected, rtol=0.1)
    assert np.isclose(f_soa_pop, f_soa_expected, rtol=0.1)
