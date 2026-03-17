# optics/refractive_index.py
from dataclasses import dataclass
from typing import Callable, Optional
from functools import lru_cache

import numpy as np
from scipy import interpolate
from ..data import open_dataset
# from ..data_old import species_open

import re

@dataclass
class RefractiveIndex:
    """Wavelength-dependent refractive index container for one species."""
    # Prefer callable access (functions) but also keep sampled arrays for debugging
    real_ri_fun: Callable[[np.ndarray], np.ndarray]
    imag_ri_fun: Callable[[np.ndarray], np.ndarray]
    wvls: Optional[np.ndarray] = None
    real_ris: Optional[np.ndarray] = None
    imag_ris: Optional[np.ndarray] = None
    RI_params: Optional[dict] = None
    

def get_RI_params(name: str) -> dict:
    n = name.upper()
    if n in ['SO4','NH4','NO3','NA','CL','MSA','CO3']:
        return {'n_550':1.55, 'k_550':0.0,  'alpha_n':0.044, 'alpha_k':0.0}
    elif n == 'BC':
        return {'n_550':1.82, 'k_550':0.74, 'alpha_n':0.0,   'alpha_k':0.0}
    elif n == 'OIN':
        return {'n_550':1.68, 'k_550':0.006,'alpha_n':0.0,   'alpha_k':0.0}
    else:  # organics default
        return {'n_550':1.45, 'k_550':0.0,  'alpha_n':0.0,   'alpha_k':0.0}



def _to_float(s: str) -> float:
    """
    Robustly parse numbers that may use Unicode scientific notation, e.g. '1.1×10−7'.
    Normalizes to standard '1.1e-7' then casts to float.
    """
    if s is None:
        raise ValueError("Cannot parse None as float")

    # strip whitespace and thin spaces
    s = str(s).strip()
    s = s.replace("\u2009", "").replace("\u202f", "").replace(" ", "")

    # normalize unicode minus and multiplication signs
    s = s.replace("\u2212", "-")   # Unicode minus
    s = s.replace("\u2013", "-")   # en dash (just in case)
    s = s.replace("\u00d7", "x")   # multiplication sign ×
    s = s.replace("×", "x")        # some fonts map separately

    # Convert patterns like '1.1x10-7' or '1.1*10-7' or '1.1X10-7' -> '1.1e-7'
    s = re.sub(r'([+-]?\d+(?:\.\d+)?)[xX\*]10([+-]?\d+)', r'\1e\2', s)

    # Convert caret form '1.1*10^-7' -> '1.1e-7'
    s = re.sub(r'([+-]?\d+(?:\.\d+)?)[xX\*]10\^([+-]?\d+)', r'\1e\2', s)

    # Also handle plain '10^...' with implied 1*10^... : '10^-3' -> '1e-3'
    s = re.sub(r'^10\^([+-]?\d+)$', r'1e\1', s)

    # final cleanup: leading/trailing plus
    s = s.lstrip("+")
    return float(s)


@lru_cache(maxsize=1)
def _load_water_ri():
    wl = []; n = []; k = []
    with open_dataset("species_data/ri_water.csv") as fh:
        for line in fh:
            if 'Wavelength' in line: 
                continue
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            w_raw, n_raw, k_raw = parts[0], parts[3], parts[4]
            wl.append(_to_float(w_raw) * 1e-6)
            n.append(_to_float(n_raw))
            k.append(_to_float(k_raw))
    wl = np.asarray(wl); n = np.asarray(n); k = np.asarray(k)
    f_n = interpolate.interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
    f_k = interpolate.interp1d(wl, k, bounds_error=False, fill_value="extrapolate")
    return f_n, f_k


def _pwr(lam, v550, alpha):
    lam = np.asarray(lam)
    return v550 * (lam / 550e-9)**alpha


def build_refractive_index(spec, wvl_grid, modifications=None):
    """Attach a RefractiveIndex to `spec` that matches the provided `wvl_grid`."""
    modifications = modifications or {}
    name = spec.name.upper()

    if name == 'H2O':
        f_n, f_k = _load_water_ri()
        real = f_n(wvl_grid)
        imag = f_k(wvl_grid)
        spec.refractive_index = RefractiveIndex(
            wvls=wvl_grid, real_ris=real, imag_ris=imag,
            real_ri_fun=f_n, imag_ri_fun=f_k, RI_params=None
        )
        return spec

    params = get_RI_params(name).copy()
    for k in ('n_550','k_550','alpha_n','alpha_k'):
        if k in modifications:
            params[k] = modifications[k]
    f_n = lambda lam: _pwr(lam, params['n_550'], params['alpha_n'])
    f_k = lambda lam: _pwr(lam, params['k_550'], params['alpha_k'])
    spec.refractive_index = RefractiveIndex(
        wvls=wvl_grid,
        real_ris=f_n(wvl_grid),
        imag_ris=f_k(wvl_grid),
        real_ri_fun=f_n,
        imag_ri_fun=f_k,
        RI_params=params
    )
    return spec


# # optics/refractive_index.py
# from functools import lru_cache
# import numpy as np
# from scipy import interpolate
# from .. import data_path

# def get_RI_params(name: str) -> dict:
#     n = name.upper()
#     if n in ['SO4','NH4','NO3','NA','CL','MSA','CO3']:
#         return {'n_550':1.55, 'k_550':0.0,  'alpha_n':0.044, 'alpha_k':0.0}
#     elif n == 'BC':
#         return {'n_550':1.82, 'k_550':0.74, 'alpha_n':0.0,   'alpha_k':0.0}
#     elif n == 'OIN':
#         return {'n_550':1.68, 'k_550':0.006,'alpha_n':0.0,   'alpha_k':0.0}
#     else:  # organics default
#         return {'n_550':1.45, 'k_550':0.0,  'alpha_n':0.0,   'alpha_k':0.0}

# @lru_cache(maxsize=1)
# def _load_water_ri(specdata_dir):
#     fn = (specdata_dir / 'ri_water.csv')
#     wl, n, k = [], [], []
#     with open(fn) as f:
#         for line in f:
#             if 'Wavelength' in line: continue
#             w, *_cols, rn, rk = line.strip().split(',')
#             wl.append(float(w) * 1e-6)  # µm → m (original had 1e-6 multiplier)
#             n.append(float(rn)); k.append(float(rk))
#     wl = np.asarray(wl); n = np.asarray(n); k = np.asarray(k)
#     return interpolate.interp1d(wl, n, bounds_error=False, fill_value="extrapolate"), \
#            interpolate.interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

# def _pwr(lam, v550, alpha):
#     return v550 * (np.asarray(lam) / 550e-9)**alpha

# def build_refractive_index(spec, wvl_grid, modifications=None, specdata_path=data_path / 'species_data'):
#     modifications = modifications or {}
#     name = spec.name.upper()
#     if name == 'H2O':
#         f_n, f_k = _load_water_ri(specdata_path)
#         real = f_n(wvl_grid)
#         imag = f_k(wvl_grid)
#         spec.refractive_index = RefractiveIndex(
#             wvls=wvl_grid, real_ris=real, imag_ris=imag,
#             real_ri_fun=f_n, imag_ri_fun=f_k, RI_params=None
#         )
#         return spec

#     params = get_RI_params(name).copy()
#     for k in ('n_550','k_550','alpha_n','alpha_k'):
#         if k in modifications: params[k] = modifications[k]

#     f_n = lambda lam: _pwr(lam, params['n_550'], params['alpha_n'])
#     f_k = lambda lam: _pwr(lam, params['k_550'], params['alpha_k'])

#     spec.refractive_index = RefractiveIndex(
#         wvls=wvl_grid,
#         real_ris=f_n(wvl_grid),
#         imag_ris=f_k(wvl_grid),
#         real_ri_fun=f_n,
#         imag_ri_fun=f_k,
#         RI_params=params
#     )
#     return spec


