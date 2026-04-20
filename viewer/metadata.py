"""Metadata helpers for the part2pop Streamlit viewer."""

from __future__ import annotations

from typing import Any, Dict, List

from part2pop.analysis.defaults import get_defaults_for_variable


STATE_LINE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "Nccn": {
        "type": "supersat",
        "axes": ["s_grid"],
        "s_range": (0.01, 10.0),
        "s_points": 100,
        "default_T": 298.15,
        "notes": "Supersaturation curve",
    },
    "frac_ccn": {
        "type": "supersat",
        "axes": ["s_grid"],
        "s_range": (0.01, 10.0),
        "s_points": 100,
        "default_T": 298.15,
        "notes": "CCN activation fraction",
    },
    "avg_Jhet": {"type": "temperature", "default_T": 298.15},
    "nucleating_sites": {"type": "temperature", "default_T": 273.15},
    "frozen_frac": {"type": "temperature", "default_T": 273.15},
    "b_abs": {
        "type": "optics",
        "morphology_options": ["core-shell", "homogeneous", "fractal"],
        "default_morphology": "core-shell",
        "rh_range": (0.0, 1.0),
        "wvl_range": (350e-9, 1150e-9),
        "rh_points": 5,
        "wvl_points": 10,
    },
    "b_scat": {
        "type": "optics",
        "morphology_options": ["core-shell", "homogeneous", "fractal"],
        "default_morphology": "core-shell",
        "rh_range": (0.0, 1.0),
        "wvl_range": (350e-9, 1150e-9),
        "rh_points": 5,
        "wvl_points": 10,
    },
    "b_ext": {
        "type": "optics",
        "morphology_options": ["core-shell", "homogeneous", "fractal"],
        "default_morphology": "core-shell",
        "rh_range": (0.0, 1.0),
        "wvl_range": (350e-9, 1150e-9),
        "rh_points": 5,
        "wvl_points": 10,
    },
    "dNdlnD": {
        "type": "distribution",
        "method_options": ["hist", "kde"],
        "N_bins_range": (20, 200),
        "D_min": 1e-9,
        "D_max": 2e-6,
        "default_method": "hist",
        "notes": "Size distribution vs. diameter",
    },
}


POPULATION_METADATA: Dict[str, Dict[str, Any]] = {
    "monodisperse": {
        "label": "Monodisperse (inline inputs)",
        "fields": [
            {"name": "N", "label": "Number concentrations", "widget": "text", "default": "1e3"},
            {"name": "D", "label": "Diameter (microns)", "widget": "text", "default": "0.1"},
            {"name": "species", "label": "Species names", "widget": "text", "default": "BC, OC"},
            {"name": "fracs", "label": "Species fractions", "widget": "text", "default": "0.1, 0.9"},
        ],
    },
    "binned_lognormals": {
        "label": "Binned lognormal",
        "fields": [
            {"name": "N", "label": "Total concentration", "widget": "number", "default": 1e4},
            {"name": "GMD", "label": "Geometric mean diameter (m)", "widget": "number", "default": 0.15e-6},
            {"name": "GSD", "label": "Geometric std dev", "widget": "number", "default": 1.4},
            {"name": "N_bins", "label": "Number of bins", "widget": "number", "default": 30},
            {"name": "species", "label": "Species names", "widget": "text", "default": "BC, OC"},
            {"name": "fracs", "label": "Species fractions", "widget": "text", "default": "0.1, 0.9"},
        ],
    },
    "partmc": {
        "label": "PartMC output",
        "config_modes": ["inline", "config_file"],
        "fields": [
            {"name": "partmc_dir", "label": "PARTMC output directory", "widget": "text", "default": "."},
            {"name": "timestep", "label": "Timestep index", "widget": "number", "default": 1, "int": True, "min": 1, "step": 1},
            {"name": "repeat", "label": "Repeat index", "widget": "number", "default": 1, "int": True, "min": 1, "step": 1},
            {"name": "n_particles", "label": "Particles to sample", "widget": "number", "default": 1000, "int": True, "min": 1, "step": 1},
            {"name": "particle_selection", "label": "Particle sampling", "widget": "select", "default": "all", "options": ["all", "sub-select"]},
        ],
        "config_file_label": "PartMC JSON config",
    },
    "mam4": {
        "label": "MAM4 namelist",
        "config_modes": ["inline", "config_file"],
        "fields": [
            {"name": "mam4_dir", "label": "MAM4 directory", "widget": "text", "default": "."},
            {"name": "timestep", "label": "Timestep (>=1)", "widget": "number", "default": 1},
            {"name": "N_bins", "label": "Number of bins", "widget": "number", "default": 20},
            {"name": "GSD", "label": "GSD list", "widget": "number_list", "default": [1.3, 1.5, 1.5, 1.5]},
            {"name": "GMD_init", "label": "Initial GMDs (m)", "widget": "number_list", "default": [1.1e-7, 2.6e-8, 2e-6, 5e-8]},
            {"name": "D_is_wet", "label": "Diameters wet", "widget": "select", "default": "True", "options": ["True", "False"]},
        ],
        "config_file_label": "MAM4 JSON config",
    },
    "hiscale_observations": {
        "label": "HI-SCALE observations",
        "config_modes": ["inline", "config_file"],
        "fields": [
            {"name": "aimms_file", "label": "AIMMS file", "widget": "text", "default": "path/to/aimms.dat"},
            {"name": "splat_file", "label": "miniSPLAT file", "widget": "text", "default": "path/to/splat.txt"},
            {"name": "ams_file", "label": "AMS file", "widget": "text", "default": "path/to/ams.dat"},
            {"name": "fims_file", "label": "FIMS file", "widget": "text", "default": "path/to/fims.dat"},
            {"name": "fims_bins_file", "label": "FIMS bins file", "widget": "text", "default": "path/to/bins.txt"},
            {"name": "z", "label": "Altitude (m)", "widget": "number", "default": 1000},
            {"name": "dz", "label": "Altitude window (m)", "widget": "number", "default": 100},
            {"name": "splat_species", "label": "miniSPLAT species mapping", "widget": "json", "default": {"BC": ["soot"], "OIN": ["Dust"]}},
            {"name": "mass_thresholds", "label": "Mass thresholds", "widget": "json", "default": {"BC": [[0.0, 1e-16, 1e-17], ["BC"]], "OIN": [[0.0, 1e-16, 1e-17], ["OIN"]]}},
        ],
        "config_file_label": "HI-SCALE JSON config",
    },
}


def get_variable_metadata(varname: str) -> Dict[str, Any]:
    """Return metadata merged with defaults for a state_line variable."""

    entry = STATE_LINE_VARIABLES.get(varname, {})
    defaults = get_defaults_for_variable(varname)
    merged = {**entry, "defaults": defaults}
    return merged


def list_state_line_variables() -> List[str]:
    """Return the list of supported state_line variables."""
    return list(STATE_LINE_VARIABLES)


def get_population_metadata() -> Dict[str, Dict[str, Any]]:
    """Return the population metadata catalog."""
    return POPULATION_METADATA


def list_population_types() -> List[str]:
    """Return the available population types in the metadata catalog."""
    return list(POPULATION_METADATA)
