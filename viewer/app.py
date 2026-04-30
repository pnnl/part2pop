"""Streamlit viewer for part2pop populations and plots."""

from pathlib import Path
import sys
import json
import re

from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import streamlit as st

VIEWER_ROOT = Path(__file__).resolve().parent
SRC_ROOT = VIEWER_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2pop.population.builder import PopulationBuilder
from part2pop.viz.builder import PlotBuilder
from part2pop.population.factory import registry as pop_registry
from part2pop.viz.factory import registry as viz_registry
from .metadata import list_population_types, list_state_line_variables
from part2pop.analysis.defaults import get_defaults_for_variable
from part2pop.analysis.particle import describe_particle_variable, list_particle_variables
from .ui import render_population_controls, render_var_controls


STATE_LINE_VARS = list_state_line_variables()


def parse_float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
        except Exception:
            return value
        if isinstance(converted, (list, tuple)):
            return [_normalize_value(item) for item in converted]
        return _normalize_value(converted)
    return value


def _merge_dicts(*dicts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for cfg in dicts:
        for key, value in cfg.items():
            merged[key] = _normalize_value(value)
    return merged


def _parse_number_list(text: str, fallback: Iterable[float]) -> list[float]:
    tokens = re.split(r"[\s,;]+", text.strip())
    values: list[float] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError:
            return list(fallback)
    return values if values else list(fallback)


def _render_config_controls(prefix: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in base_cfg.items():
        widget_key = f"{prefix}_{key}"
        if isinstance(value, bool):
            resolved[key] = st.checkbox(key, value=value, key=widget_key)
        elif isinstance(value, (int, float)):
            resolved[key] = st.number_input(key, value=float(value), key=widget_key)
        elif isinstance(value, str):
            resolved[key] = st.text_input(key, value=value, key=widget_key)
        elif isinstance(value, Iterable):
            display = ", ".join(map(str, value))
            text = st.text_input(key, value=display, key=widget_key)
            resolved[key] = _parse_number_list(text, value)
        else:
            resolved[key] = value
    return resolved


def _collect_particle_var_cfg(xvar: str | None, yvar: str | None) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    metadata_list: list[Dict[str, Any]] = []
    merged_defaults: Dict[str, Any] = {}
    axis_keys: list[str] = []
    for var in (xvar, yvar):
        if not var:
            continue
        meta = describe_particle_variable(var)
        metadata_list.append(meta)
        merged_defaults = _merge_dicts(merged_defaults, meta.get("defaults", {}))
        axis_keys.extend([axis for axis in meta.get("axis_keys", []) if axis])
    axis_defaults: Dict[str, Any] = {}
    for axis in axis_keys:
        axis_defaults.update(get_defaults_for_variable(axis))
    merged_defaults = _merge_dicts(axis_defaults, merged_defaults)
    return metadata_list, merged_defaults


def _normalize_numeric_key_dict(value: Any) -> Any:
    if isinstance(value, dict):
        digit_keys = [k for k in value if isinstance(k, str) and k.isdigit()]
        if digit_keys and len(digit_keys) == len(value):
            sorted_keys = sorted(digit_keys, key=lambda key: int(key))
            return [_normalize_numeric_key_dict(value[key]) for key in sorted_keys]
        return {key: _normalize_numeric_key_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_numeric_key_dict(item) for item in value]
    return value


def parse_species_list(text: str) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()]


def parse_population_field(cfg: dict, field: str, default: str) -> list[float]:
    return parse_float_list(cfg.get(field, default))


def load_config_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        st.error(f"Failed to load config file '{path}': {exc}")
        return {}


def finalize_population_config(raw: dict) -> dict:
    cfg = _normalize_numeric_key_dict(dict(raw))
    config_path = cfg.pop("config_file", None)
    if config_path:
        file_cfg = load_config_file(config_path)
        if file_cfg:
            return _normalize_numeric_key_dict(file_cfg)
        st.warning("Using inline entries after failing to load config file.")
    pop_type = cfg.get("type")
    if pop_type == "monodisperse":
        try:
            return {
                "type": "monodisperse",
                "N": parse_population_field(cfg, "N", "1e3"),
                "D": parse_population_field(cfg, "D", "0.1"),
                "aero_spec_names": [parse_species_list(cfg.get("species", "BC, OC"))],
                "aero_spec_fracs": [parse_population_field(cfg, "fracs", "0.1, 0.9")],
            }
        except ValueError as exc:
            st.error(f"Invalid monodisperse values: {exc}")
            return {}
    if pop_type == "binned_lognormals":
        base = {
            "type": "binned_lognormals",
            "N": [float(cfg.get("N", 1e4))],
            "GMD": [float(cfg.get("GMD", 0.15e-6))],
            "GSD": [float(cfg.get("GSD", 1.4))],
            "N_bins": int(cfg.get("N_bins", 30)),
            "aero_spec_names": [parse_species_list(cfg.get("species", "BC, OC"))],
            "aero_spec_fracs": [parse_population_field(cfg, "fracs", "0.1, 0.9")],
        }
        extras = {key: val for key, val in cfg.items() if key not in base}
        base.update(extras)
        return base
    if pop_type in ("partmc", "mam4"):
        return {key: val for key, val in cfg.items() if val is not None}
    return _normalize_numeric_key_dict(cfg)


def build_population_options() -> list[str]:
    return list(pop_registry.discover_population_types().keys())


def build_plot_options() -> list[str]:
    return list(viz_registry.discover_plotter_types().keys())


def _sanitize_hiscale_config(cfg: Dict[str, Any]) -> None:
    if "splat_species" in cfg:
        cfg["splat_species"] = _normalize_numeric_key_dict(cfg["splat_species"])
    if "mass_thresholds" in cfg:
        cfg["mass_thresholds"] = _normalize_numeric_key_dict(cfg["mass_thresholds"])


def run_viewer() -> None:
    st.set_page_config(page_title="part2pop sandbox viewer", layout="wide")
    st.title("part2pop Sandbox Viewer")

    population_types = build_population_options()
    plot_types = build_plot_options()

    with st.sidebar:
        st.header("Population configuration")
        population_type = st.selectbox("Population type", population_types)
        raw_population_cfg = render_population_controls(population_type)
        population_config = finalize_population_config(raw_population_cfg)
        if population_config.get("type") == "hiscale_observations":
            _sanitize_hiscale_config(population_config)

        st.header("Visualization")
        plot_type = st.selectbox("Plot type", plot_types)
        xvar = yvar = None
        state_line_var = None
        var_cfg: Dict[str, Any] = {}
        scales = {"xscale": "linear", "yscale": "linear"}
        scatter_metadata: list[Dict[str, Any]] = []
        if plot_type == "state_scatter":
            scatter_vars = list_particle_variables()
            if scatter_vars:
                default_y_index = 1 if len(scatter_vars) > 1 else 0
                xvar = st.selectbox("State scatter X variable", scatter_vars, index=0, key="state_scatter_xvar")
                yvar = st.selectbox(
                    "State scatter Y variable",
                    scatter_vars,
                    index=default_y_index,
                    key="state_scatter_yvar",
                )
            scatter_metadata, merged_defaults = _collect_particle_var_cfg(xvar, yvar)
            overrides = _render_config_controls("state_scatter", merged_defaults)
            var_cfg = _merge_dicts(merged_defaults, overrides)
            scales["xscale"] = st.selectbox("X axis scale", ["linear", "log"], index=0, key="state_scatter_xscale")
            scales["yscale"] = st.selectbox("Y axis scale", ["linear", "log"], index=0, key="state_scatter_yscale")
        else:
            state_line_var = st.selectbox("State line variable", STATE_LINE_VARS)
            var_cfg = render_var_controls(state_line_var)
        st.button("Refresh plot")
        show_diagnostics = st.checkbox("Show diagnostics", value=False)

    if not population_config:
        st.warning("Define a population configuration to render the plot.")
        return

    dump_path = VIEWER_ROOT / "population_config_dump.json"
    with dump_path.open("w", encoding="utf-8") as fp:
        json.dump(population_config, fp, indent=2)

    try:
        with st.spinner("Building population..."):
            population = PopulationBuilder(population_config).build()
            if plot_type == "state_scatter":
                plot_config = {
                    "xvar": xvar,
                    "yvar": yvar,
                    "var_cfg": var_cfg,
                    **scales,
                }
            else:
                plot_config = {"varname": state_line_var, "var_cfg": var_cfg}
            plotter = PlotBuilder(plot_type, plot_config).build()

        fig, ax = plt.subplots()
        plotter.plot(population, ax)
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Failed to render plot: {exc}")
        return

    
    if show_diagnostics:
        st.markdown("### Population stats")
        st.write("Total concentration", float(population.get_Ntot()))
        st.write("Total dry mass", float(population.get_tot_dry_mass()))

        with st.expander("Diagnostics", expanded=True):
            st.write("population_config", population_config)
            st.write("state_line_var", state_line_var)
            st.write("var_cfg", var_cfg)


if __name__ == "__main__":
    run_viewer()