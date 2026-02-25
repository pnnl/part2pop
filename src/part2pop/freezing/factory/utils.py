#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:25:36 2025

@author: beel083
"""
import numpy as np
  
def calculate_Psat(T):
    """
    Murphy & Koop (2005) saturation vapor pressures (Pa).

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    (p_ice, p_liq) : tuple of floats
        p_ice : saturation vapor pressure over planar ice (Pa)
        p_liq : saturation vapor pressure over planar liquid water (Pa)

    Notes
    -----
    Common validity ranges cited for these fits:
      - Ice:   ~110 K to 273.16 K
      - Liquid: ~123 K to 332 K (includes supercooled liquid)
    
    Formulas (natural log, pressure in Pa):
      ln(p_ice) = 9.550426 - 5723.265/T + 3.53068*ln(T) - 0.00728332*T
      ln(p_liq) = 54.842763 - 6763.22/T - 4.210*ln(T) + 0.000367*T
                  + tanh(0.0415*(T-218.8)) *
                    (53.878 - 1331.22/T - 9.44523*ln(T) + 0.014025*T)
    """
    if T <= 0:
        raise ValueError("Temperature must be in Kelvin and > 0.")

    lnT = np.log(T)

    # Saturation vapor pressure over ice (Pa)
    ln_p_ice = 9.550426 - (5723.265 / T) + 3.53068 * lnT - 0.00728332 * T
    p_ice = np.exp(ln_p_ice)
    
    # Saturation vapor pressure over liquid water (Pa)
    tanh_term = np.tanh(0.0415 * (T - 218.8))
    ln_p_liq = (
        54.842763
        - (6763.22 / T)
        - 4.210 * lnT
        + 0.000367 * T
        + tanh_term
        * (
            53.878
            - (1331.22 / T)
            - 9.44523 * lnT
            + 0.014025 * T
        )
    )
    p_liq = np.exp(ln_p_liq)

    return p_liq, p_ice
    

def calculate_dPsat_dT(T):
    """
    Murphy & Koop (2005) dP/dT.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin

    Returns
    -------
    dp_dT_ice : Pa/K
    dp_dT_liq : Pa/K
    """
    T = np.asarray(T)
    lnT = np.log(T)

    # -------------------
    # ICE
    # -------------------
    ln_p_ice = (
        9.550426
        - 5723.265 / T
        + 3.53068 * lnT
        - 0.00728332 * T
    )

    p_ice = np.exp(ln_p_ice)

    dlnp_dT_ice = (
        5723.265 / T**2
        + 3.53068 / T
        - 0.00728332
    )

    dp_dT_ice = p_ice * dlnp_dT_ice

    # -------------------
    # LIQUID WATER
    # -------------------
    x = 0.0415 * (T - 218.8)
    tanh_x = np.tanh(x)
    sech2_x = 1.0 - tanh_x**2  # sech^2(x)

    A = (
        54.842763
        - 6763.22 / T
        - 4.210 * lnT
        + 0.000367 * T
    )

    B = (
        53.878
        - 1331.22 / T
        - 9.44523 * lnT
        + 0.014025 * T
    )

    ln_p_liq = A + tanh_x * B
    p_liq = np.exp(ln_p_liq)

    # derivatives
    A_prime = (
        6763.22 / T**2
        - 4.210 / T
        + 0.000367
    )

    B_prime = (
        1331.22 / T**2
        - 9.44523 / T
        + 0.014025
    )

    dlnp_dT_liq = (
        A_prime
        + tanh_x * B_prime
        + 0.0415 * sech2_x * B
    )

    dp_dT_liq = p_liq * dlnp_dT_liq

    return dp_dT_liq, dp_dT_ice

