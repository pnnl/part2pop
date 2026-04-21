import numpy as np
import pytest
from part2pop import build_population, build_optical_population
from part2pop._patch import patch_pymiescatt

@pytest.mark.requires_pymiescatt
@pytest.mark.integration
def test_optical_properties_with_pymiescatt():
    # Patch PyMieScatt for compatibility and import Mie_Lognormal
    patch_pymiescatt()
    from PyMieScatt import Mie_Lognormal

    # Define a single lognormal population (N:1000 cm^-3, GMD:100 nm, GSD:1.6, sulfate)
    pop_cfg = {
        "type": "binned_lognormals",
        "N": [1e9],              # 1e9 m^-3 = 1000 cm^-3
        "GMD": [100e-9],         # 100 nm geometric mean diameter
        "GSD": [1.6],
        "N_sigmas": 6, 
        "N_bins": 200,          # use 200 size bins for discretization
        "aero_spec_names": [[ "SO4" ]],
        "aero_spec_fracs": [[1.0]],
        "species_modifications": { "SO4": {"n_550": 1.45, "k_550": 0.0} }  # refractive index at 550 nm
    }
    pop = build_population(pop_cfg)

    # Compute optical coefficients with part2pop (homogeneous sphere morphology)
    opt_cfg = {"rh_grid": [0.0], "wvl_grid": [550e-9], "type": "homogeneous", "temp": 298.15}
    opt_pop = build_optical_population(pop, opt_cfg)
    b_scat = opt_pop.get_optical_coeff("b_scat", rh=0.0, wvl=550e-9)  # scattering (m^-1)
    b_abs  = opt_pop.get_optical_coeff("b_abs",  rh=0.0, wvl=550e-9)  # absorption (m^-1)
    
    # Compute optical coefficients with PyMieScatt for the same lognormal distribution
    refr_index = complex(1.45, 0.0)   # refractive index (real=1.45, imag=0 for sulfate)
    wavelength_nm = 550.0            # nm
    gsd = 1.6
    gmd_nm = 100.0                   # nm
    N0_cm3 = 1000.0                  # 1000 cm^-3
    out = Mie_Lognormal(refr_index, wavelength_nm, gsd, gmd_nm, N0_cm3,
                        lower=1e9*pop.get_particle(1).get_Dwet(), upper=1e9*pop.get_particle(pop_cfg['N_bins']).get_Dwet(), asDict=True, numberOfBins=200)
    # Extract scattering (Bsca) and absorption (Babs) from PyMieScatt output (in Mm^-1)
    bsca_Mm = out.get("Bsca")# or out.get("Bsca, Mm^-1") or out.get("Bsca (Mm^-1)")
    babs_Mm = out.get("Babs")# or out.get("Babs, Mm^-1") or out.get("Babs (Mm^-1)")
    assert bsca_Mm is not None and babs_Mm is not None, "PyMieScatt output missing Bsca/Babs"
    # Convert PyMieScatt output to SI units (m^-1)
    bsca_SI = float(bsca_Mm) * 1e-6
    babs_SI = float(babs_Mm) * 1e-6

    # Assert that scattering and absorption match within 2% relative tolerance
    assert np.isclose(b_scat, bsca_SI, rtol=0.02), f"Scattering mismatch: {b_scat} vs {bsca_SI}"
    assert np.isclose(b_abs,  babs_SI, rtol=0.02), f"Absorption mismatch: {b_abs} vs {babs_SI}"
