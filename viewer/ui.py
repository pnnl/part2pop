"""UI helpers for Streamlit controls in the part2pop viewer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st

from .metadata import POPULATION_METADATA, get_variable_metadata


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


def slider_grid(label: str, lo: float, hi: float, points: int) -> List[float]:
    return list(np.linspace(lo, hi, num=points))


def parse_number_list(text: str, fallback: List[float]) -> List[float]:
    cleaned = ",".join(map(str, fallback)) if not text else text
    tokens = re.split(r"[\s,;]+", cleaned.strip())
    values: List[float] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError:
            st.warning(f"Could not parse '{tok}' as a float; using fallback grid.")
            return list(fallback)
    if not values:
        return list(fallback)
    return values


def guess_partmc_final_timestep(partmc_dir: str) -> int:
    try:
        out_dir = Path(partmc_dir or ".") / "out"
        if not out_dir.is_dir():
            return 1
        candidate = 1
        pattern = re.compile(r"_(\d{4})_(\d{8})\.nc$")
        for entry in out_dir.iterdir():
            if not entry.is_file():
                continue
            match = pattern.search(entry.name)
            if not match:
                continue
            timestep = int(match.group(2))
            candidate = max(candidate, timestep)
        return candidate
    except Exception:
        return 1


def render_var_controls(varname: str) -> Dict[str, Any]:
    meta = get_variable_metadata(varname)
    defaults: Dict[str, Any] = meta.get("defaults", {})
    cfg: Dict[str, Any] = dict(defaults)

    st.subheader(f"Variable: {varname}")
    if meta.get("type") == "supersat":
        s_range = tuple(float(val) for val in meta.get("s_range", (0.01, 10.0)))
        if len(s_range) < 2:
            s_range = (s_range[0], s_range[0])
        slider_result = st.slider(
            "Supersaturation range",
            min_value=s_range[0],
            max_value=s_range[1],
            value=(s_range[0], s_range[1]),
            key=f"{varname}_srange",
        )
        if isinstance(slider_result, tuple):
            lo, hi = slider_result
        else:
            lo = slider_result
            hi = slider_result
        points = st.slider("Points", meta.get("s_points", 20), 400, value=meta.get("s_points", 100), key=f"{varname}_spoints")
        cfg["s_grid"] = slider_grid("s_grid", lo, hi, points)
        cfg["s_eval"] = cfg["s_grid"]
        cfg.setdefault("T", meta.get("default_T", 298.15))
    elif meta.get("type") == "temperature":
        t_range = tuple(float(val) for val in defaults.get("T_range", (273.15, 258.15)))
        if len(t_range) < 2:
            t_range = (t_range[0], t_range[0])
        slider_res = st.slider(
            "Temperature range (K)",
            min_value=t_range[1],
            max_value=t_range[0],
            value=(t_range[0], t_range[1]),
            key=f"{varname}_trange",
        )
        if isinstance(slider_res, tuple):
            t_lo, t_hi = slider_res
        else:
            t_lo = slider_res
            t_hi = slider_res
        points = st.slider("Points", 5, 200, value=defaults.get("T_points", 20), key=f"{varname}_Tpoints")
        cfg["T_grid"] = slider_grid("T_grid", t_hi, t_lo, points)
        cfg["cooling_rate"] = st.number_input("Cooling rate (K/s)", value=float(defaults.get("cooling_rate", 0.1)), key=f"{varname}_cooling")
        cfg.setdefault("T_units", defaults.get("T_units", "K"))
    elif meta.get("type") == "optics":
        cfg.setdefault("rh_grid", slider_grid("RH grid", *meta.get("rh_range", (0.0, 1.0)), meta.get("rh_points", 5)))
        cfg.setdefault("wvl_grid", slider_grid("Wavelength grid", *meta.get("wvl_range", (350e-9, 1150e-9)), meta.get("wvl_points", 6)))
        cfg.setdefault("wvls", list(cfg["wvl_grid"]))
        cfg.setdefault("T", defaults.get("T", 298.15))
        morphology = st.selectbox("Morphology", meta.get("morphology_options", [meta.get("default_morphology")]), index=0, key=f"{varname}_morph")
        cfg["morphology"] = morphology
        cfg.setdefault("species_modifications", {})
        cfg["vary_rh"] = st.checkbox("Vary RH", value=defaults.get("vary_rh", True), key=f"{varname}_vary_rh")
        cfg["vary_wvl"] = st.checkbox("Vary wavelength", value=defaults.get("vary_wvl", True), key=f"{varname}_vary_wvl")
    elif varname == "dNdlnD":
        method = st.selectbox("Method", meta.get("method_options", [meta.get("default_method")]), index=0, key="dNdlnD_method")
        cfg["method"] = method
        cfg.setdefault("N_bins", st.slider("Bins", *meta.get("N_bins_range", (20, 200)), value=80, key="dNdlnD_bins"))
        cfg.setdefault("D_min", meta.get("D_min", 1e-9))
        cfg.setdefault("D_max", meta.get("D_max", 2e-6))
        cfg.setdefault("normalize", False)
        cfg.setdefault("wetsize", defaults.get("wetsize", True))
    return cfg


def render_population_controls(pop_type: str) -> Dict[str, Any]:
    meta = POPULATION_METADATA.get(pop_type, {})
    cfg: Dict[str, Any] = {"type": pop_type}
    st.subheader(f"Population: {meta.get('label', pop_type)}")
    use_config = False
    if meta.get("config_modes"):
        modes = meta["config_modes"]
        use_config = st.radio("Configuration mode", modes, index=0, key=f"{pop_type}_mode") == "config_file"
    if use_config:
        cfg_path = st.text_input(meta.get("config_file_label", "Config file"), key=f"{pop_type}_file")
        cfg["config_file"] = cfg_path
        return cfg

    for field in meta.get("fields", []):
        widget = field["widget"]
        label = field["label"]
        default = field.get("default")
        key = f"{pop_type}_{field['name']}"
        if widget == "number":
            is_int = field.get("int", False)
            default_val = default if default is not None else (1 if is_int else 0)
            if is_int:
                cfg[field["name"]] = st.number_input(
                    label,
                    value=int(default_val),
                    min_value=field.get("min"),
                    step=field.get("step", 1),
                    key=key,
                    format="%d",
                )
            else:
                cfg[field["name"]] = st.number_input(
                    label,
                    value=float(default_val),
                    min_value=field.get("min"),
                    step=field.get("step", 0.1),
                    key=key,
                )
        elif widget == "select":
            options = field.get("options", [])
            cfg[field["name"]] = st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)
        elif widget == "json":
            text_val = json.dumps(default, indent=2) if default is not None else ""
            user_input = st.text_area(label, value=text_val, key=key)
            try:
                parsed = json.loads(user_input) if user_input else {}
                cfg[field["name"]] = _normalize_numeric_key_dict(parsed)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON for {field['name']}: {exc}")
                cfg[field["name"]] = default or {}
        elif widget == "number_list":
            values = default or []
            display = ", ".join(map(str, values))
            user_input = st.text_input(label, value=display, key=key)
            cfg[field["name"]] = [float(v) for v in re.split(r"[\s,;]+", user_input.strip()) if v]
        else:
            cfg[field["name"]] = st.text_input(label, value=str(default or ""), key=key)
    return cfg
