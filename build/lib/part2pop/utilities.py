"""Small utilities used by part2pop.

This module provides helpers for parsing numeric strings and computing
power moments of lognormal size distributions used by population builders
and examples.
"""

import numpy as np


def get_number(string_val):
    """Parse a numeric string into a float.

    Accepts plain floats or strings using a multiplication sign and exponent
    notation (e.g. "1.23×10^3").

    Parameters
    ----------
    string_val : str
        Numeric string to parse.

    Returns
    -------
    float
        Parsed numeric value.
    """
    import re

    # Normalize and trim
    if isinstance(string_val, str):
        s = string_val.strip()
    else:
        # accept numeric inputs directly
        return float(string_val)

    # Replace common unicode variants
    s = s.replace("\u00d7", "x").replace("×", "x")
    s = s.replace("\u2212", "-")  # unicode minus
    s = s.replace("\u2013", "-")  # en-dash
    s = s.replace("\u2009", "").replace("\u202f", "")
    s = s.replace(" ", "")

    # Convert 'x10^', 'x10', '*10^', etc. into 'e' scientific notation
    # Examples handled: '1.2x10-3', '1.2x10^3', '3x10^2', '3×10^2'
    s2 = re.sub(r"[xX\*]10\^?", "e", s)
    # handle leading '10^...' -> '1e...'
    s2 = re.sub(r'^10\^', '1e', s2)

    try:
        return float(s2)
    except Exception:
        # final fallback: try plain float on original cleaned string
        try:
            return float(s)
        except Exception:
            raise ValueError(f"Cannot parse numeric string: {string_val}")


def power_moments_from_lognormal(k, N, gmd, gsd):
    """Compute the k-th power moment for one lognormal mode.

    Parameters
    ----------
    k : int
        Moment order.
    N : float
        Number concentration prefactor.
    gmd : float
        Geometric mean diameter.
    gsd : float
        Geometric standard deviation.

    Returns
    -------
    float
        k-th power moment value.
    """
    return N * np.exp(k * np.log(gmd) + k ** 2 * np.log(gsd) / 2.0)


def power_moments_from_lognormals(k, Ns, GMDs, GSDs):
    """Sum k-th power moments across multiple lognormal modes.

    Parameters mirror `power_moments_from_lognormal` but accept sequences.
    """
    return np.sum([
        power_moments_from_lognormal(k, N, gmd, gsd)
        for (N, gmd, gsd) in zip(Ns, GMDs, GSDs)
    ])