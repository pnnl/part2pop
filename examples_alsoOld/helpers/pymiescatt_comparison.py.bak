from typing import Dict, Any
import numpy as np
from scipy.stats import norm

# Compatibility: patch PyMieScatt if the project provides a patch hook
try:
    from PyParticle._patch import patch_pymiescatt
    patch_pymiescatt()
    from PyMieScatt import AutoMieQ, Mie_Lognormal, Mie_SD, MieQ, MieQCoreShell
except Exception as exc:  # pragma: no cover - visible in runtime only
    raise ModuleNotFoundError(
        "Install PyMieScatt to run reference Mie comparisons: pip install PyMieScatt"
    ) from exc

_MMINV_TO_MINV = 1e-6  # Mm^-1 -> m^-1
_M_TO_NM = 1e9
_CM3_to_M3 = 1e6

# def reference_optics_for_population(
#     pop_cfg: Dict[str, Any],
#     var_cfg: Dict[str, Any],
#     *,
#     wvl_units: str = "m",
#     output_units: str = "m^-1",
#     enforce_single_species: bool = True,
# ) -> Dict[str, np.ndarray]:
#     """
#     Compute reference Mie optics for a single-species lognormal population using PyMieScatt.

#     Returns SI-consistent arrays:
#       - 'wvl'    : np.ndarray, wavelengths in meters [m]
#       - 'b_scat' : np.ndarray, scattering coefficient (m^-1 by default)
#       - 'b_abs'  : np.ndarray, absorption coefficient (m^-1 by default)

#     Notes
#     -----
#     - var_cfg['wvl_grid'] is expected in meters by default; set wvl_units='nm' if you pass nm.
#     - RH handling: supports only a single RH value (e.g., [0.0]).
#     - Supports single-species populations (provide 'species_modifications' with 'n_550' and 'k_550').
#     """
#     if var_cfg is None:
#         var_cfg = {}

#     # --- wavelength normalization: produce SI meters output ---
#     if "wvl_grid" not in var_cfg:
#         raise ValueError("var_cfg must include 'wvl_grid' (wavelengths).")
#     wvl_arr = np.asarray(var_cfg["wvl_grid"], dtype=float)
#     if wvl_units == "nm":
#         wvl_m = wvl_arr * 1e-9
#     elif wvl_units == "m":
#         wvl_m = wvl_arr
#     else:
#         raise ValueError("wvl_units must be 'm' or 'nm'.")

#     # Only support single RH for this simple reference helper
#     rh_grid = var_cfg.get("rh_grid", [0.0])
#     if len(rh_grid) != 1:
#         raise ValueError(
#             "reference_optics_for_population currently supports a single RH value (e.g., [0.0])."
#         )

#     # --- population validation and parameter extraction ---
#     mods = pop_cfg.get("species_modifications")
#     if not mods or not isinstance(mods, dict):
#         raise ValueError("pop_cfg must include 'species_modifications' with refractive index entries.")
    
#     if enforce_single_species and len(mods) != 1:
#         raise ValueError("reference helper requires a single species in 'species_modifications' for now.")
    
#     species_name = list(mods.keys())[0]
#     spec_mods = mods[species_name] or {}
#     try:
#         n_550 = float(spec_mods["n_550"])
#         k_550 = float(spec_mods["k_550"])
#     except KeyError as e:
#         raise ValueError("species_modifications must include 'n_550' and 'k_550' for the species.") from e
#     refr = complex(n_550, k_550)

#     gmd = pop_cfg.get("GMD")
#     gsd = pop_cfg.get("GSD")
#     N0 = pop_cfg.get("N")
    
#     if len(gmd)>1 or len(gsd)>1 or len(N0)>1:
#         raise ValueError("reference_optics_for_population currently supports single-mode populations only.")
#     gmd = gmd[0]
#     gsd = gsd[0]
#     N0 = N0[0]
#     n_bins = int(pop_cfg.get("N_bins", 400))

#     # Convert GMD to nm for PyMieScatt
#     gmd_units = pop_cfg.get("GMD_units", "m")
#     if gmd_units == "m":
#         gmd_nm = gmd * 1e9
#     elif gmd_units in ("nm", "nanometer", "nanometers"):
#         gmd_nm = gmd
#     else:
#         raise ValueError("Unsupported GMD_units. Use 'm' or 'nm'.")

#     # Convert number concentration to cm^-3 for PyMieScatt
#     n_units = pop_cfg.get("N_units", "m-3")
#     if n_units in ("m-3", "m^-3"):
#         N0_cm3 = N0 / 1e6
#     elif n_units in ("cm-3", "cm^-3"):
#         N0_cm3 = N0
#     else:
#         raise ValueError("Unsupported N_units. Use 'm-3' or 'cm-3'.")
    
#     print(gmd_nm, gmd, N0_cm3, N0)
#     # diameter integration bounds (convert to nm for PyMieScatt)
#     dmin = pop_cfg.get("D_min")#, None)
#     dmax = pop_cfg.get("D_max")#, None)

#     def _to_nm(d):
#         if d is None:
#             return None
#         if gmd_units == "m":
#             return float(d) * 1e9
#         return float(d)
    
#     lower_nm = _to_nm(dmin) #if dmin is not None else None
#     upper_nm = _to_nm(dmax)# if dmax is not None else None

#     # if lower_nm is None:
#     #     lower_nm = gmd_nm / 20.0
#     # if upper_nm is None:
#     #     upper_nm = gmd_nm * 20.0
    
#     # Convert wavelengths to nm for PyMieScatt calls
#     wvl_nm_list = wvl_m * 1e9 #list((wvl_m * 1e9).astype(float))

#     b_scat_Mm = []
#     b_abs_Mm = []
#     for wl_nm in wvl_nm_list:
#         print(refr,wl_nm,gsd,gmd_nm,N0_cm3,lower_nm,upper_nm,n_bins)
#         #out = Mie_SD()
#         out = Mie_Lognormal(
#             refr,
#             wl_nm,
#             gsd,
#             gmd_nm,
#             N0_cm3,
#             lower=lower_nm,
#             upper=upper_nm,
#             asDict=True,
#             numberOfBins=n_bins,
#         )
#         # robust key lookup
#         if isinstance(out, dict):
#             bsca = out.get("Bsca") or out.get("Bsca, Mm^-1") or out.get("Bsca (Mm^-1)")
#             babs = out.get("Babs") or out.get("Babs, Mm^-1") or out.get("Babs (Mm^-1)")
#         else:
#             bsca = out[0] if len(out) > 0 else 0.0
#             babs = out[1] if len(out) > 1 else 0.0
#         b_scat_Mm.append(float(bsca or 0.0))
#         b_abs_Mm.append(float(babs or 0.0))
    
#     b_scat_Mm = np.asarray(b_scat_Mm, dtype=float)
#     b_abs_Mm = np.asarray(b_abs_Mm, dtype=float)

#     # Convert to desired output units
#     if output_units == "m^-1":
#         factor = _MMINV_TO_MINV
#     elif output_units == "Mm^-1":
#         factor = 1.0
#     else:
#         raise ValueError("output_units must be 'm^-1' or 'Mm^-1'.")

#     b_scat_out = b_scat_Mm * factor
#     b_abs_out = b_abs_Mm * factor

#     return {
#         "wvl": np.asarray(wvl_m, dtype=float),
#         "b_scat": b_scat_out,
#         "b_abs": b_abs_out,
#         "meta": {
#             "wvl_units": "m",
#             "output_units": output_units,
#             "n_bins": n_bins,
#             "species": species_name,
#         },
#     }


# # def pymiescatt_lognormal_optics(pop_cfg, var_cfg):
# #     """
# #     Backwards-compatible wrapper matching previous helper behavior.
# #     Returns: (wvl_nm_array, b_scat_m, b_abs_m)
# #     """
# #     d = reference_optics_for_population(pop_cfg, var_cfg, wvl_units="m", output_units="m^-1")
# #     # Return wavelengths in nm to preserve older notebook behavior
# #     return (np.asarray(d["wvl"]) * 1e9, d["b_scat"], d["b_abs"])
# from typing import Tuple
# import numpy as np
# import warnings

# try:
#     from PyParticle._patch import patch_pymiescatt
#     patch_pymiescatt()
#     #import PyMieScatt as PMS
#     from PyMieScatt import Mie_Lognormal
# except Exception as e:
#     raise ModuleNotFoundError("Install PyMieScatt to run direct Mie comparison: pip install PyMieScatt") from e

# MMINVERSE_TO_MINVERSE = 1e-6  # 1 / (1 Mm) = 1e-6 1/m
# M_TO_NM = 1e9                 # meters -> nanometers (for PyMieScatt interface)

# # todo: fix this
# def pymiescatt_lognormal_optics(pop_cfg, var_cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute Mie optics for a (possibly multi-modal) lognormal aerosol using PyMieScatt,
#     returning units that match the rest of PyParticle.

#     Parameters
#     ----------
#     pop_cfg : dict
#         Population config. This helper does **not** alter your existing parsing logic
#         (refractive index, modes, etc.). It assumes you've already built the parameters
#         needed for PyMieScatt internally.
#     var_cfg : dict
#         Must include 'wvl_grid' **in meters** (SI), as used across PyParticle.

#     Returns
#     -------
#     wvl_nm : np.ndarray
#         Wavelengths in **nanometers** (kept for backward compatibility with the notebook,
#         which multiplies by 1e-9 before plotting).
#     b_scat_m : np.ndarray
#         Scattering coefficient in **m⁻¹** (SI).  PyMieScatt returns Mm⁻¹; we convert here.
#     b_abs_m : np.ndarray
#         Absorption coefficient in **m⁻¹** (SI).  PyMieScatt returns Mm⁻¹; we convert here.

#     Notes
#     -----
#     PyMieScatt’s API uses nm for wavelength/diameter and cm⁻³ for concentrations,
#     and returns coefficients in Mm⁻¹. See docs. We only standardize the *outputs*
#     (to m⁻¹) here so the rest of the package stays SI-consistent.
#     """
#     # --- your existing logic begins (kept as-is) ---
#     # Expect that somewhere below you:
#     #   - read var_cfg["wvl_grid"] (meters)
#     #   - build wavelength(s) for PyMieScatt in nm
#     #   - call PyMieScatt and obtain Bsca/Babs in Mm^-1
#     #
#     # To keep this patch minimal and risk-free, we don't change how you build inputs.
#     # We only enforce output unit conversions right before returning.

#     # Handle missing var_cfg gracefully
#     if var_cfg is None:
#         var_cfg = {}
    
#     wvl_m = np.asarray(var_cfg.get("wvl_grid", [550e-9]), dtype=float)
#     if wvl_m.ndim != 1:
#         wvl_m = wvl_m.reshape(-1)
#     wvl_nm = wvl_m * M_TO_NM

#     # Helpers
#     def _first(x):
#         if isinstance(x, (list, tuple)):
#             return x[0]
#         return x

#     # Extract modal parameters with safe fallbacks
#     gmd = pop_cfg.get('GMD')[0]#float(_first(pop_cfg.get('GMD', pop_cfg.get('gmd', 0.0))))
#     gsd = pop_cfg.get('GSD')[0]#float(_first(pop_cfg.get('GSD', pop_cfg.get('gsd', 1.6))))
#     N0 = pop_cfg.get('N')[0]#loat(_first(pop_cfg.get('N', pop_cfg.get('N0', pop_cfg.get('N_tot', 1000.0)))))

#     # Convert GMD to nm for PyMieScatt
#     gmd_units = pop_cfg.get('GMD_units', 'm')
#     if gmd_units == 'm':
#         dg_nm = gmd * M_TO_NM
#     elif gmd_units in ('nm', 'nanometer', 'nanometers'):
#         dg_nm = gmd
#     else:
#         raise ValueError(f"Unsupported GMD_units: {gmd_units}. Supported: 'm', 'nm'")
    
#     # Convert number concentration to cm^-3 for PyMieScatt
#     n_units = pop_cfg.get('N_units','m-3')
#     if n_units in ('m-3', 'm^-3'):
#         N0_cm3 = N0 / 1e6
#     elif n_units in ('cm-3', 'cm^-3'):
#         N0_cm3 = N0
#     else:
#         raise ValueError(f"Unsupported N_units: {n_units}. Supported: 'm-3', 'cm-3'")
    
#     # Determine the species-level modifications for the single-species case.
#     # species_modifications is expected to be a dict keyed by species name, e.g.
#     # {"SO4": {"n_550":1.45, "k_550":0.0}}
#     all_spec_mods = pop_cfg['species_modifications']#if 'species_modifications' in pop_cfg else {}
#     # all_spec_mods = pop_cfg.get("species_modifications", {})# or {}
#     # Prefer explicit single-key in species_modifications
#     if len(all_spec_mods) == 1:
#         species_name = list(all_spec_mods.keys())[0]
#         spec_mods = all_spec_mods.get(species_name, {}) if species_name else {}
#     else:
#         raise NotImplementedError("PyMieScatt comparison currently only supports single-species populations.")
    
#     ri_real = spec_mods['n_550']# if 'n_550' in spec_mods else 1.5
#     ri_imag = spec_mods['k_550']# if 'k_550' in spec_mods else 0.0
    
#     if spec_mods.get('alpha_n', 0.0) != 0.0 or spec_mods.get('alpha_k', 0.0) != 0.0:
#         warnings.warn("Population-level spectral slope (alpha_n, alpha_k) not supported in PyMieScatt comparison; using n_550/k_550 only.")
#     refr = complex(float(ri_real), float(ri_imag))
    
#     #var_cfg.get('refractive_index', var_cfg.get('m', 1.5 + 0.0j)))
#     if isinstance(refr, (list, tuple)) and len(refr) >= 2:
#         try:
#             n_val = float(refr[0])
#             k_val = float(refr[1])
#             refr = complex(n_val, k_val)
#         except Exception:
#             refr = complex(float(refr[0]), 0.0)
    
#     # wavelength list for PyMieScatt (nm)
#     wl_nm_list = list(wvl_nm)

#     # If the population config provides D_min/D_max, prefer those bounds (convert to nm)
#     dmin = pop_cfg.get('D_min', 1e-9) # fallback to 1 nm
#     dmax = pop_cfg.get('D_max', 1e-2) # fallback to 1 cm
#     gmd_units = pop_cfg.get('GMD_units', 'm')

#     if gmd_units == 'm':
#         lower = dmin * M_TO_NM
#         upper = dmax * M_TO_NM
#     elif gmd_units in ('nm', 'nanometer', 'nanometers'):
#         lower = dmin
#         upper = dmax
#     else:
#         raise ValueError(f"Unsupported D_min_units: {gmd_units}. Supported: 'm', 'nm'")

#     N_bins = pop_cfg['N_bins'] if 'N_bins' in pop_cfg else 400
    
#     diam_grid_nm = np.logspace(np.log10(lower), np.log10(upper), N_bins)
    
#     dlogD = np.log10(diam_grid_nm[1]) - np.log10(diam_grid_nm[0])
    
#     w = N0_cm3*norm(loc=np.log10(dg_nm), scale=np.log10(gsd)).pdf(np.log10(diam_grid_nm))

#     b_scat_Mm1 = []
#     b_abs_Mm1 = []
#     for wl in wl_nm_list:
#         #output_dict = Mie_SD(refr, wl, diam_grid_nm, w, asDict=True, SMPS=False)
        
        
#         output_dict = Mie_Lognormal(
#             refr,
#             wl,
#             gsd,
#             dg_nm, # in nm
#             N0_cm3,
#             lower=lower, # in nm
#             upper=upper, # in nm
#             asDict=True,
#             numberOfBins=pop_cfg['N_bins'])
        
#         #Bext, Bsca, Babs, G, Bpr, Bback, Bratio
#         # output_dict = Mie_Lognormal(
#         #     refr,
#         #     wl,
#         #     gsd,
#         #     dg_nm,
#         #     N0_cm3,
#         #     lower=lower,
#         #     upper=upper,
#         #     asDict=True,
#         #     numberOfBins=pop_cfg['N_bins'],
#         #     SMPS=False)
#         Bsca = output_dict['Bsca']
#         Babs = output_dict['Babs']
#         # def _get_key(dct, keys):
#         #     for k in keys:
#         #         if k in dct:
#         #             return dct[k]
#         #     return None
#         bsca_raw = Bsca
#         babs_raw = Babs
#         # if isinstance(out, dict):
#         #     bsca_raw = _get_key(out, ['Bsca', 'Bsca, Mm^-1', 'Bsca, Mm^-1', 'Bsca (Mm^-1)'])
#         #     babs_raw = _get_key(out, ['Babs', 'Babs, Mm^-1', 'Babs (Mm^-1)'])
#         # else:
#         #     try:
#         #         bsca_raw = out[0]
#         #     except Exception:
#         #         bsca_raw = 0.0
#         #     try:
#         #         babs_raw = out[1]
#         #     except Exception:
#         #         babs_raw = 0.0

#         bsca_raw = float(bsca_raw) if bsca_raw is not None else 0.0
#         babs_raw = float(babs_raw) if babs_raw is not None else 0.0

#         b_scat_Mm1.append(bsca_raw)
#         b_abs_Mm1.append(babs_raw)

#     b_scat_Mm1 = np.asarray(b_scat_Mm1, dtype=float)
#     b_abs_Mm1 = np.asarray(b_abs_Mm1, dtype=float)

#     # Convert to m^-1
#     b_scat_m = b_scat_Mm1 * MMINVERSE_TO_MINVERSE
#     b_abs_m = b_abs_Mm1 * MMINVERSE_TO_MINVERSE

#     return np.asarray(wl_nm_list), b_scat_m, b_abs_m



from typing import Tuple
import numpy as np
import warnings

try:
    from PyParticle._patch import patch_pymiescatt
    patch_pymiescatt()
    #import PyMieScatt as PMS
    from PyMieScatt import Mie_Lognormal
except Exception as e:
    raise ModuleNotFoundError("Install PyMieScatt to run direct Mie comparison: pip install PyMieScatt") from e

MMINVERSE_TO_MINVERSE = 1e-6  # 1 / (1 Mm) = 1e-6 1/m
M_TO_NM = 1e9                 # meters -> nanometers (for PyMieScatt interface)

# todo: fix this
def pymiescatt_lognormal_optics(pop_cfg, var_cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Mie optics for a (possibly multi-modal) lognormal aerosol using PyMieScatt,
    returning units that match the rest of PyParticle.

    Parameters
    ----------
    pop_cfg : dict
        Population config. This helper does **not** alter your existing parsing logic
        (refractive index, modes, etc.). It assumes you've already built the parameters
        needed for PyMieScatt internally.
    var_cfg : dict
        Must include 'wvl_grid' **in meters** (SI), as used across PyParticle.

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
    # --- your existing logic begins (kept as-is) ---
    # Expect that somewhere below you:
    #   - read var_cfg["wvl_grid"] (meters)
    #   - build wavelength(s) for PyMieScatt in nm
    #   - call PyMieScatt and obtain Bsca/Babs in Mm^-1
    #
    # To keep this patch minimal and risk-free, we don't change how you build inputs.
    # We only enforce output unit conversions right before returning.

    # Handle missing var_cfg gracefully
    if var_cfg is None:
        var_cfg = {}
    
    wvl_m = np.asarray(var_cfg.get("wvl_grid", [550e-9]), dtype=float)
    if wvl_m.ndim != 1:
        wvl_m = wvl_m.reshape(-1)
    wvl_nm = wvl_m * M_TO_NM

    # Helpers
    def _first(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    # Extract modal parameters with safe fallbacks
    gmd = pop_cfg.get('GMD')[0]#float(_first(pop_cfg.get('GMD', pop_cfg.get('gmd', 0.0))))
    gsd = pop_cfg.get('GSD')[0]#float(_first(pop_cfg.get('GSD', pop_cfg.get('gsd', 1.6))))
    N0 = pop_cfg.get('N')[0]#loat(_first(pop_cfg.get('N', pop_cfg.get('N0', pop_cfg.get('N_tot', 1000.0)))))

    # Convert GMD to nm for PyMieScatt
    gmd_units = pop_cfg.get('GMD_units', 'm')
    if gmd_units == 'm':
        dg_nm = gmd * M_TO_NM
    elif gmd_units in ('nm', 'nanometer', 'nanometers'):
        dg_nm = gmd
    else:
        raise ValueError(f"Unsupported GMD_units: {gmd_units}. Supported: 'm', 'nm'")
    
    # Convert number concentration to cm^-3 for PyMieScatt
    n_units = pop_cfg.get('N_units','m-3')
    if n_units in ('m-3', 'm^-3'):
        N0_cm3 = N0 / 1e6
    elif n_units in ('cm-3', 'cm^-3'):
        N0_cm3 = N0
    else:
        raise ValueError(f"Unsupported N_units: {n_units}. Supported: 'm-3', 'cm-3'")
    
    # Determine the species-level modifications for the single-species case.
    # species_modifications is expected to be a dict keyed by species name, e.g.
    # {"SO4": {"n_550":1.45, "k_550":0.0}}
    all_spec_mods = pop_cfg['species_modifications']#if 'species_modifications' in pop_cfg else {}
    # all_spec_mods = pop_cfg.get("species_modifications", {})# or {}
    # Prefer explicit single-key in species_modifications
    if len(all_spec_mods) == 1:
        species_name = list(all_spec_mods.keys())[0]
        spec_mods = all_spec_mods.get(species_name, {}) if species_name else {}
    else:
        raise NotImplementedError("PyMieScatt comparison currently only supports single-species populations.")
    
    
    ri_real = spec_mods['n_550']# if 'n_550' in spec_mods else 1.5
    ri_imag = spec_mods['k_550']# if 'k_550' in spec_mods else 0.0
    
    
    # ri_real = spec_mods.get('n_550', 1.5)
    # ri_imag = spec_mods.get('k_550', 0.)
    if spec_mods.get('alpha_n', 0.0) != 0.0 or spec_mods.get('alpha_k', 0.0) != 0.0:
        warnings.warn("Population-level spectral slope (alpha_n, alpha_k) not supported in PyMieScatt comparison; using n_550/k_550 only.")
    refr = complex(float(ri_real), float(ri_imag))
    
    #var_cfg.get('refractive_index', var_cfg.get('m', 1.5 + 0.0j)))
    if isinstance(refr, (list, tuple)) and len(refr) >= 2:
        try:
            n_val = float(refr[0])
            k_val = float(refr[1])
            refr = complex(n_val, k_val)
        except Exception:
            refr = complex(float(refr[0]), 0.0)
    
    # wavelength list for PyMieScatt (nm)
    wl_nm_list = list(wvl_nm)

    # If the population config provides D_min/D_max, prefer those bounds (convert to nm)
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

    # Fall back to safe defaults if neither provided
    if lower is None:
        lower = dg_nm / 20.0 #if gmd > 0 else 1.0
    if upper is None:
        upper = dg_nm * 20.0 #if gmd > 0 else 1000.0
    
    N_bins = pop_cfg['N_bins'] #if 'N_bins' in pop_cfg else 400
    
    diam_grid_nm = np.logspace(np.log10(lower), np.log10(upper), N_bins)
    
    # dlogD = np.log10(diam_grid_nm[1]) - np.log10(diam_grid_nm[0])
    
    dlogD = np.log10(diam_grid_nm[1]) - np.log10(diam_grid_nm[0])
    dlnD =  np.log(diam_grid_nm[1]) - np.log(diam_grid_nm[0])
    N_per_bin = dlogD*N0_cm3*norm(loc=np.log10(dg_nm), scale=np.log10(gsd)).pdf(np.log10(diam_grid_nm))
    print(np.sum(N_per_bin), N0_cm3)
    N_per_bin_cm3 = N_per_bin/np.sum(N_per_bin)*N0_cm3

    b_scat_Mm1 = []
    b_abs_Mm1 = []
    for wl in wl_nm_list:
        #output_dict = Mie_SD(refr, wl, diam_grid_nm, w, asDict=True, SMPS=False)
        
        
        # output_dict = Mie_Lognormal(
        #     refr,
        #     wl,
        #     gsd,
        #     dg_nm, # in nm
        #     N0_cm3,
        #     lower=lower, # in nm
        #     upper=upper, # in nm
        #     asDict=True,
        #     numberOfBins=pop_cfg['N_bins'])
        
        Babs = 0.
        Bsca = 0.
        Bext = 0.

        Cext = np.zeros(N_bins)
        Csca = np.zeros(N_bins)
        Cabs = np.zeros(N_bins)
        for ii,(d, N_per_cm3) in enumerate(zip(diam_grid_nm, N_per_bin_cm3)):
            if N_per_cm3 > 0:
                N_per_m3 = N_per_cm3 * 1e6 # cm^-3 to m^-3
                output_dict = AutoMieQ(refr, wl, d, asCrossSection=False, asDict=True)
                Qext = output_dict['Qext']
                Qsca = output_dict['Qsca']
                Qabs = output_dict['Qabs']
                Cext[ii] = Qext * np.pi * (d/2/_M_TO_NM)**2 # m^2
                Csca[ii] = Qsca * np.pi * (d/2/_M_TO_NM)**2 # m^2
                Cabs[ii] = Qabs * np.pi * (d/2/_M_TO_NM)**2 # m^2
        Bext = np.sum(Cext * N_per_bin_cm3 * 1e6) # m^-1 
        Bsca = np.sum(Csca * N_per_bin_cm3 * 1e6) # m^-1 
        Babs = np.sum(Cabs * N_per_bin_cm3 * 1e6) # m^-1 
        # Bsca = np.sum(output_dict['Qsca'] * np.pi * (diam_grid_nm/2)**2 * w)
        # Babs = np.sum(output_dict['Qabs'] * np.pi * (diam_grid_nm/2)**2 * w)
        
        bsca_raw = Bsca
        babs_raw = Babs
        # bsca_raw = float(bsca_raw) if bsca_raw is not None else 0.0
        # babs_raw = float(babs_raw) if babs_raw is not None else 0.0
        
        b_scat_Mm1.append(bsca_raw / MMINVERSE_TO_MINVERSE)
        b_abs_Mm1.append(babs_raw / MMINVERSE_TO_MINVERSE)

    b_scat_Mm1 = np.asarray(b_scat_Mm1, dtype=float)
    b_abs_Mm1 = np.asarray(b_abs_Mm1, dtype=float)

    # Convert to m^-1
    b_scat_m = b_scat_Mm1 * MMINVERSE_TO_MINVERSE
    b_abs_m = b_abs_Mm1 * MMINVERSE_TO_MINVERSE

    return np.asarray(wl_nm_list), b_scat_m, b_abs_m
