from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import norm
import warnings

# Compatibility: patch PyMieScatt if the project provides a patch hook
try:
    from part2pop._patch import patch_pymiescatt
    patch_pymiescatt()
    from PyMieScatt import AutoMieQ, Mie_Lognormal, Mie_SD, MieQ, MieQCoreShell
except Exception as exc:  # pragma: no cover - visible in runtime only
    raise ModuleNotFoundError(
        "Install PyMieScatt to run reference Mie comparisons: pip install PyMieScatt"
    ) from exc

_MMINV_TO_MINV = 1e-6  # Mm^-1 -> m^-1
_M_TO_NM = 1e9
_CM3_to_M3 = 1e6

try:
    from part2pop._patch import patch_pymiescatt
    patch_pymiescatt()
    #import PyMieScatt as PMS
    from PyMieScatt import Mie_Lognormal
except Exception as e:
    raise ModuleNotFoundError("Install PyMieScatt to run direct Mie comparison: pip install PyMieScatt") from e

MMINVERSE_TO_MINVERSE = 1e-6  # 1 / (1 Mm) = 1e-6 1/m
M_TO_NM = 1e9                 # meters -> nanometers (for PyMieScatt interface)

def pymiescatt_lognormal_optics(pop_cfg, var_cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Mie optics for a (possibly multi-modal) lognormal aerosol using PyMieScatt,
    returning units that match the rest of part2pop.

    Parameters
    ----------
    pop_cfg : dict
        Population config. This helper does **not** alter your existing parsing logic
        (refractive index, modes, etc.). It assumes you've already built the parameters
        needed for PyMieScatt internally.
    var_cfg : dict
        Must include 'wvl_grid' **in meters** (SI), as used across part2pop.

    Returns
    -------
    wvl_nm : np.ndarray
        Wavelengths in **nanometers** (kept for backward compatibility with the notebook,
        which multiplies by 1e-9 before plotting).
    b_scat_m : np.ndarray
        Scattering coefficient in **m⁻¹** (SI).  PyMieScatt returns Mm⁻¹; we convert here.
    b_abs_m : np.ndarray
        Absorption coefficient in **m⁻¹** (SI).  PyMieScatt returns Mm⁻¹; we convert here.

    Notes
    -----
    PyMieScatt’s API uses nm for wavelength/diameter and cm⁻³ for concentrations,
    and returns coefficients in Mm⁻¹. See docs. We only standardize the *outputs*
    (to m⁻¹) here so the rest of the package stays SI-consistent.
    """
    if var_cfg is None:
        var_cfg = {}
    
    wvl_m = np.asarray(var_cfg.get("wvl_grid", [550e-9]), dtype=float)
    if wvl_m.ndim != 1:
        wvl_m = wvl_m.reshape(-1)
    wvl_nm = wvl_m * M_TO_NM

    def _first(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    gmd = pop_cfg.get('GMD')[0]
    gsd = pop_cfg.get('GSD')[0]
    N0 = pop_cfg.get('N')[0]

    gmd_units = pop_cfg.get('GMD_units', 'm')
    if gmd_units == 'm':
        dg_nm = gmd * M_TO_NM
    elif gmd_units in ('nm', 'nanometer', 'nanometers'):
        dg_nm = gmd
    else:
        raise ValueError(f"Unsupported GMD_units: {gmd_units}. Supported: 'm', 'nm'")
    
    n_units = pop_cfg.get('N_units','m-3')
    if n_units in ('m-3', 'm^-3'):
        N0_cm3 = N0 / 1e6
    elif n_units in ('cm-3', 'cm^-3'):
        N0_cm3 = N0
    else:
        raise ValueError(f"Unsupported N_units: {n_units}. Supported: 'm-3', 'cm-3'")
    
    all_spec_mods = pop_cfg['species_modifications']
    if len(all_spec_mods) == 1:
        species_name = list(all_spec_mods.keys())[0]
        spec_mods = all_spec_mods.get(species_name, {}) if species_name else {}
    else:
        raise NotImplementedError("PyMieScatt comparison currently only supports single-species populations.")
    
    ri_real = spec_mods['n_550']
    ri_imag = spec_mods['k_550']
    if spec_mods.get('alpha_n', 0.0) != 0.0 or spec_mods.get('alpha_k', 0.0) != 0.0:
        warnings.warn("Population-level spectral slope (alpha_n, alpha_k) not supported in PyMieScatt comparison; using n_550/k_550 only.")
    refr = complex(float(ri_real), float(ri_imag))
    
    if isinstance(refr, (list, tuple)) and len(refr) >= 2:
        try:
            n_val = float(refr[0])
            k_val = float(refr[1])
            refr = complex(n_val, k_val)
        except Exception:
            refr = complex(float(refr[0]), 0.0)
    
    wl_nm_list = list(wvl_nm)

    dmin = pop_cfg.get('D_min', None)
    dmax = pop_cfg.get('D_max', None)
    if dmin is not None:
        if gmd_units == 'm':
            lower = float(dmin) * M_TO_NM
        elif gmd_units in ('nm', 'nanometer', 'nanometers'):
            lower = float(dmin)
        else:
            lower = float(dmin) * M_TO_NM
    if dmax is not None:
        if gmd_units == 'm':
            upper = float(dmax) * M_TO_NM
        elif gmd_units in ('nm', 'nanometer', 'nanometers'):
            upper = float(dmax)
        else:
            upper = float(dmax) * M_TO_NM

    if lower is None:
        lower = dg_nm / 20.0
    if upper is None:
        upper = dg_nm * 20.0
    
    N_bins = pop_cfg['N_bins']
    
    diam_grid_nm = np.logspace(np.log10(lower), np.log10(upper), N_bins)
    
    dlogD = np.log10(diam_grid_nm[1]) - np.log10(diam_grid_nm[0])
    N_per_bin = dlogD*N0_cm3*norm(loc=np.log10(dg_nm), scale=np.log10(gsd)).pdf(np.log10(diam_grid_nm))
    N_per_bin_cm3 = N_per_bin/np.sum(N_per_bin)*N0_cm3

    b_scat_Mm1 = []
    b_abs_Mm1 = []
    for wl in wl_nm_list:
        Babs = 0.
        Bsca = 0.
        Bext = 0.

        Cext = np.zeros(N_bins)
        Csca = np.zeros(N_bins)
        Cabs = np.zeros(N_bins)
        for ii,(d, N_per_cm3) in enumerate(zip(diam_grid_nm, N_per_bin_cm3)):
            if N_per_cm3 > 0:
                output_dict = AutoMieQ(refr, wl, d, asCrossSection=False, asDict=True)
                Qext = output_dict['Qext']
                Qsca = output_dict['Qsca']
                Qabs = output_dict['Qabs']
                Cext[ii] = Qext * np.pi * (d/2/_M_TO_NM)**2 # m^2
                Csca[ii] = Qsca * np.pi * (d/2/_M_TO_NM)**2 # m^2
                Cabs[ii] = Qabs * np.pi * (d/2/_M_TO_NM)**2 # m^2
        Bext = np.sum(Cext * N_per_bin_cm3 * 1e6)
        Bsca = np.sum(Csca * N_per_bin_cm3 * 1e6)
        Babs = np.sum(Cabs * N_per_bin_cm3 * 1e6)
        
        bsca_raw = Bsca
        babs_raw = Babs

        b_scat_Mm1.append(bsca_raw / MMINVERSE_TO_MINVERSE)
        b_abs_Mm1.append(babs_raw / MMINVERSE_TO_MINVERSE)

    b_scat_Mm1 = np.asarray(b_scat_Mm1, dtype=float)
    b_abs_Mm1 = np.asarray(b_abs_Mm1, dtype=float)

    b_scat_m = b_scat_Mm1 * MMINVERSE_TO_MINVERSE
    b_abs_m = b_abs_Mm1 * MMINVERSE_TO_MINVERSE

    return np.asarray(wl_nm_list), b_scat_m, b_abs_m
