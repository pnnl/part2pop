from __future__ import annotations
import numpy as np
from typing import Dict, Any

"""Simple defaults provider for analysis variables.

This module exposes a single function `get_defaults_for_variable(name)` which
returns a dict of defaults appropriate for the requested variable name. The
function centralizes common axis/grid defaults and provides a single place to
add more defaults for other variables.

Rationale: callers (e.g., the VariableBuilder) can ask for defaults by
variable name and merge them with variable-local defaults and user config.
"""

# fixme: maybe associate this with variables

# A compact mapping of canonical defaults keyed by variable name. The map is
# intentionally permissive: axis variables and consuming variables share the
# same entries so a single lookup returns the canonical grids for both.
_DEFAULTS_BY_VAR: Dict[str, Dict[str, Any]] = {
    # supersaturation-related
    "s_grid": {
        "s_grid": np.logspace(-2, 1, 100),
        "s_eval": np.logspace(-2, 1, 100),
    },
    "Nccn": {
        "s_grid": np.logspace(-2, 1, 100),
        "s_eval": np.logspace(-2, 1, 100),
        "T": 298.15,
    },
    "frac_ccn": {
        "s_grid": np.logspace(-2, 1, 100),
        "s_eval": np.logspace(-2, 1, 100),
        "T": 298.15,
    },

    # fixme: deal with distributions more generally?
    # diameter grid / distribution defaults
    "diam_grid": {
        "diam_grid": np.logspace(-9, -6, 80),
    },
    "dNdlnD": {
        "diam_grid": np.logspace(-9, -6, 80),
        # keep distribution-related options here as a convenience; variable
        # meta.default_cfg may also set N_bins, method, etc.
        "wetsize": True,
        "normalize": False,
        "method": "hist",
        "N_bins": 80,
        "D_min": 1e-9,
        "D_max": 2e-6,
        "diam_scale": "log",
    },

    # wavelength / optics defaults
    "wvl_grid": {"wvl_grid": [550e-9], "wvls": [550e-9]},
    "b_scat": {
        "wvl_grid": [550e-9],
        "wvls": [550e-9],
        "rh_grid": [0.0, 0.5, 0.9],
        "morphology": "core-shell",
        "species_modifications": {},
        "T": 298.15,
    },
    "b_abs": {
        "wvl_grid": [550e-9],
        "wvls": [550e-9],
        "rh_grid": [0.0, 0.5, 0.9],
        "morphology": "core-shell",
        "species_modifications": {},
        "T": 298.15,
    },
    "b_ext": {
        "wvl_grid": [550e-9],
        "wvls": [550e-9],
        "rh_grid": [0.0, 0.5, 0.9],
        "morphology": "core-shell",
        "species_modifications": {},
        "T": 298.15,
    },

    # relative-humidity axis
    "rh_grid": {"rh_grid": np.asarray([0.0])},

    # other/legacy entries (keep for compatibility)
    "wetsize": {},
    "__fallback__": {},
}


def get_defaults_for_variable(name: str) -> Dict[str, Any]:
    """Return a shallow copy of defaults for the given variable name.

    The name should be the canonical variable name (e.g., 'Nccn', 's_grid',
    'b_scat'). If no defaults are known for the name, an empty dict is
    returned.
    """
    # Normalize simple aliasing: many code paths use lower-case or alias
    # forms; try direct lookup, then case-insensitive, then fallback.
    if name in _DEFAULTS_BY_VAR:
        return dict(_DEFAULTS_BY_VAR[name])
    # try some common alias forms
    for key in (name, name.lower(), name.capitalize()):
        if key in _DEFAULTS_BY_VAR:
            return dict(_DEFAULTS_BY_VAR[key])
    return dict(_DEFAULTS_BY_VAR.get("__fallback__", {}))


def all_defaults() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the whole defaults mapping (diagnostic)."""
    return {k: dict(v) for k, v in _DEFAULTS_BY_VAR.items()}

