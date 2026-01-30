#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from HI-SCALE style observational inputs:
  - FIMS: size distribution (binned N(Dp) or dN/dlnDp / dN/dlog10Dp)
  - AIMMS: aircraft state for altitude/region/time filtering
  - miniSPLAT: reduced-class number fractions
  - AMS: bulk mass fractions (SO4, NO3, OC, NH4)

Design goals:
  - Keep everything in ONE builder file (no separate io module)
  - Follow the structure of other population builders in part2pop
  - Deterministic: split each FIMS size bin into particle "types"
  - Minimal dependencies: numpy only

Config keys (required):
{
  "type": "hiscale_observations",
  "fims_file": "...",
  "fims_bins_file": "...",          # explicit lo/hi bin edges (nm)
  "aimms_file": "...",
  "splat_file": "...",
  "ams_file": "...",
  "z": 100.0,
  "dz": 2.0,
  "splat_species": { "BC": ["BC"], "OIN": ["OIN"], ... }
}

Config keys (optional):
  cloud_flag_value: int/float, default 0
  max_dp_nm: float, default 1000
  splat_cutoff_nm: float, default 0 (no cutoff)
  region_filter: dict with lon/lat bounds (defaults match separate_tools)
  composition_strategy: "ams_everywhere" | "templates_fallback_to_ams" | "templates_only"
  type_templates: dict type->dict(spec->massfrac)
  ams_species_map: dict AMS key-> part2pop species name (default {"SO4":"SO4","NO3":"NO3","OC":"OC","NH4":"NH4"})
  species_modifications: dict spec_name -> property overrides
  D_is_wet: bool, default False

  aimms_time_col/alt/lat/lon: override AIMMS column names if needed
  aimms_cloud_col: optional AIMMS cloud column (usually None)
  fims_time_col: optional FIMS time column name override
  fims_cloud_col: optional FIMS cloud column name override (default "Cloud_flag")

  fims_density_measure: "ln" | "log10" | "per_bin"
      - "ln" means FIMS bins are dN/dlnDp
      - "log10" means dN/dlog10Dp
      - "per_bin" means already per-bin N
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import re

import numpy as np

from ..base import ParticlePopulation
from ..utils import normalize_population_config
from part2pop import make_particle
from part2pop.species.registry import get_species
from .registry import register


# -----------------------------
# Generic text helpers
# -----------------------------

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def _parse_nheader_firstline(first_line: str) -> int:
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 1:
        raise ValueError(f"Bad first line (expected 'NHEADER,...'): {first_line}")
    try:
        return int(float(parts[0]))
    except Exception as e:
        raise ValueError(f"Could not parse NHEADER from first line: {first_line}") from e


def _split_csv(line: str) -> List[str]:
    return [tok.strip() for tok in line.split(",")]


def _read_icartt_table(path: str) -> Dict[str, np.ndarray]:
    """
    ICARTT-ish reader:
      - line 0: "NHEADER, NRECS" (or similar)
      - line (NHEADER-1): comma-separated column header line
      - data begins at line NHEADER
    Returns dict col->float array.
    """
    lines = _read_lines(path)
    if len(lines) < 2:
        raise ValueError(f"File too short: {path}")

    nheader = _parse_nheader_firstline(lines[0])
    if nheader <= 1 or nheader >= len(lines):
        raise ValueError(f"Parsed NHEADER={nheader} out of range for file {path} (nlines={len(lines)})")

    header_line = lines[nheader - 1]
    colnames = _split_csv(header_line)

    data_lines = lines[nheader:]
    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        rows.append(_split_csv(line))

    arr = np.array(rows, dtype="float64")
    if arr.ndim != 2:
        raise ValueError(f"Unexpected data array shape for {path}: {arr.shape}")
    if arr.shape[1] != len(colnames):
        raise ValueError(
            f"Column count mismatch in {path}: header has {len(colnames)} cols, "
            f"data has {arr.shape[1]} cols.\nHeader line: {header_line}"
        )

    out: Dict[str, np.ndarray] = {}
    for j, name in enumerate(colnames):
        out[name] = arr[:, j]
    return out


def _read_delimited_table_with_header(path: str) -> Dict[str, np.ndarray]:
    """
    Simple delimited table reader:
      - first non-empty line is header
      - remaining non-empty lines are data
    Supports tab, comma, or whitespace. Assumes numeric cells.
    """
    lines = _read_lines(path)

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Empty file: {path}")

    header_line = lines[header_idx].strip()
    if "\t" in header_line:
        delim: Optional[str] = "\t"
    elif "," in header_line:
        delim = ","
    else:
        delim = None

    if delim is None:
        headers = header_line.split()
    else:
        headers = [h.strip() for h in header_line.split(delim)]
    headers = [h for h in headers if h != ""]
    if len(headers) < 2:
        raise ValueError(f"Could not parse header columns from: {header_line}")

    rows: List[List[str]] = []
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not s:
            continue
        if delim is None:
            toks = s.split()
        else:
            toks = [t.strip() for t in s.split(delim)]
        toks = [t for t in toks if t != ""]
        if len(toks) < len(headers):
            continue
        rows.append(toks[:len(headers)])

    if not rows:
        raise ValueError(f"No data rows parsed from {path}")

    arr = np.array(rows, dtype="float64")
    if arr.shape[1] != len(headers):
        raise ValueError(f"Parsed data width mismatch in {path}: {arr.shape[1]} vs {len(headers)}")

    out: Dict[str, np.ndarray] = {}
    for j, h in enumerate(headers):
        out[h] = arr[:, j]
    return out


# -----------------------------
# AIMMS
# -----------------------------

def _read_aimms(aimms_file: str) -> Dict[str, np.ndarray]:
    return _read_icartt_table(aimms_file)


# -----------------------------
# FIMS core + bins
# -----------------------------

def _read_fims_core(fims_file: str) -> Dict[str, Any]:
    """
    Parse FIMS via ICARTT header + numeric columns.
    Detect size-bin columns by header pattern (n_Dp*, ndp*). Returns:
      - all columns
      - "_bin_cols": ordered list of bin col names
      - "_N_bins": (nt, nbins) array
    """
    table = _read_icartt_table(fims_file)
    keys = list(table.keys())

    # Common FIMS bin col patterns: n_Dp_1, n_Dp_2, ... or ndp_...
    bin_cols = [k for k in keys if re.match(r"(?i)^n[_\-]?dp", k) or re.match(r"(?i)^ndp", k)]

    # If that fails, try selecting a contiguous block after the first column (time)
    if not bin_cols and len(keys) >= 8:
        # choose the longest block starting at index 1
        # (ICARTT already gave numeric arrays for all cols, so this is mostly about intent)
        best_start = 1
        best_len = len(keys) - 1
        # require many bins to accept
        if best_len >= 10:
            bin_cols = keys[best_start:best_start + best_len]

    if not bin_cols:
        raise ValueError(
            f"Could not identify FIMS bin columns in {fims_file}. "
            f"Headers (sample): {keys[:50]}"
        )

    # preserve header order
    bin_cols = [k for k in keys if k in set(bin_cols)]
    N_bins = np.vstack([table[c] for c in bin_cols]).T.astype(float)

    out: Dict[str, Any] = dict(table)
    out["_bin_cols"] = bin_cols
    out["_N_bins"] = N_bins
    return out


def _read_fims_bins_file(fims_bins_file: str, expected_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a bins file and return (Dp_lowers_nm, Dp_uppers_nm) with length expected_bins.
    Works for:
      - 2-column tables (lower upper)
      - numeric streams that are interleaved lo,hi,lo,hi...
      - numeric streams that are concatenated all-lo then all-hi
    """
    lines = _read_lines(fims_bins_file)

    # Try per-line 2-col parse first
    lo_list: List[float] = []
    hi_list: List[float] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(vals) >= 2:
            lo_list.append(float(vals[0]))
            hi_list.append(float(vals[1]))
    if len(lo_list) == expected_bins and len(hi_list) == expected_bins:
        lo = np.array(lo_list, dtype="float64")
        hi = np.array(hi_list, dtype="float64")
        if np.all(hi > lo) and np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0):
            return lo, hi

    # Global numeric stream heuristics
    txt = "\n".join(lines)
    all_vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if len(all_vals) < 2 * expected_bins:
        raise ValueError(
            f"Bins file {fims_bins_file} does not contain enough numeric values for {expected_bins} bins."
        )

    x = np.array([float(v) for v in all_vals], dtype="float64")

    # Interleaved
    lo = x[0:2 * expected_bins:2]
    hi = x[1:2 * expected_bins:2]
    if lo.size == expected_bins and np.all(hi > lo) and np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0):
        return lo, hi

    # Concatenated
    lo = x[:expected_bins]
    hi = x[expected_bins:2 * expected_bins]
    if lo.size == expected_bins and np.all(hi > lo) and np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0):
        return lo, hi

    # Last resort: if the file contains multiple blocks, try sliding windows
    n_needed = 2 * expected_bins
    for start in range(0, len(x) - n_needed + 1):
        block = x[start:start + n_needed]

        lo_i = block[0::2]
        hi_i = block[1::2]
        if np.all(hi_i > lo_i) and np.all(np.diff(lo_i) >= 0) and np.all(np.diff(hi_i) >= 0):
            return lo_i, hi_i

        lo_c = block[:expected_bins]
        hi_c = block[expected_bins:]
        if np.all(hi_c > lo_c) and np.all(np.diff(lo_c) >= 0) and np.all(np.diff(hi_c) >= 0):
            return lo_c, hi_c

    raise ValueError(
        f"Could not extract {expected_bins} (lo,hi) bin pairs from {fims_bins_file}. "
        f"First numeric values: {x[:30]}"
    )


# -----------------------------
# Time / altitude / region / cloud selection
# -----------------------------

def _fims_time_indices_for_altitude_and_cloudflag(
    *,
    fims: Dict[str, Any],
    aimms: Dict[str, np.ndarray],
    z: float,
    dz: float,
    cloud_flag_value: Optional[float] = 0,
    region_filter: Optional[Dict[str, float]] = None,
    aimms_time_col: str = "Time(UTC)",
    aimms_alt_col: str = "Alt",
    aimms_lat_col: str = "Lat",
    aimms_lon_col: str = "Lon",
    aimms_cloud_col: Optional[str] = None,
    fims_time_col: str = "Time(UTC)",
    fims_cloud_col: str = "Cloud_flag",
) -> np.ndarray:
    """
    Returns indices into FIMS rows that correspond to AIMMS points in the requested
    altitude window (+ region) by matching rounded AIMMS times to FIMS times.

    Cloud handling:
      - AIMMS: only applied if aimms_cloud_col is provided AND exists.
      - FIMS: only applied if fims_cloud_col exists in fims.
        If it doesn't exist, we do NOT error; we simply skip that filter.
    """
    if region_filter is None:
        region_filter = {"lon_min": -97.5, "lon_max": -97.4, "lat_min": 36.05, "lat_max": 36.81}

    for k in (aimms_time_col, aimms_alt_col, aimms_lat_col, aimms_lon_col):
        if k not in aimms:
            raise KeyError(f"AIMMS missing required column '{k}'. Available: {sorted(aimms.keys())[:50]}")

    if fims_time_col not in fims:
        raise KeyError(f"FIMS missing time column '{fims_time_col}'. Available: {sorted(fims.keys())[:50]}")

    t_aimms = aimms[aimms_time_col]
    alt = aimms[aimms_alt_col]
    lat = aimms[aimms_lat_col]
    lon = aimms[aimms_lon_col]

    in_region = (
        (lon > region_filter["lon_min"]) & (lon < region_filter["lon_max"]) &
        (lat > region_filter["lat_min"]) & (lat < region_filter["lat_max"])
    )

    idx_alt = np.where((alt >= z - dz) & (alt < z + dz) & in_region)[0]
    idx_leg = np.where(in_region)[0]

    # clamp like separate_tools if altitude window empty
    if idx_alt.size == 0 and idx_leg.size > 0:
        alt_leg = alt[idx_leg]
        if (z - dz) > np.max(alt_leg):
            idx_alt = np.where((alt > (np.max(alt_leg) - dz)) & in_region)[0]
        elif (z + dz) < np.min(alt_leg):
            idx_alt = np.where((alt < (np.min(alt_leg) + dz)) & in_region)[0]

    if idx_alt.size == 0:
        return np.array([], dtype=int)

    # optional AIMMS cloud filter
    if aimms_cloud_col is not None and (aimms_cloud_col in aimms) and (cloud_flag_value is not None):
        cloud = aimms[aimms_cloud_col]
        idx_alt = idx_alt[np.where(cloud[idx_alt] == cloud_flag_value)[0]]
        if idx_alt.size == 0:
            return np.array([], dtype=int)

    # time match
    aimms_times = np.unique(np.round(t_aimms[idx_alt], 0)).astype(float)

    t_fims = np.asarray(fims[fims_time_col], dtype=float)
    mask = np.isin(t_fims, aimms_times)

    # optional FIMS cloud-flag filter (ONLY if the column exists)
    if cloud_flag_value is not None and (fims_cloud_col in fims):
        mask = mask & (np.asarray(fims[fims_cloud_col], dtype=float) == float(cloud_flag_value))

    idx = np.where(mask)[0].astype(int)
    return np.unique(idx)


# -----------------------------
# FIMS average size distribution
# -----------------------------

def _read_fims_avg_size_dist(
    *,
    fims_file: str,
    fims_bins_file: str,
    aimms_file: str,
    z: float,
    dz: float,
    cloud_flag_value: Optional[float] = 0,
    max_dp_nm: float = 1000.0,
    region_filter: Optional[Dict[str, float]] = None,
    aimms_time_col: str = "Time(UTC)",
    aimms_alt_col: str = "Alt",
    aimms_lat_col: str = "Lat",
    aimms_lon_col: str = "Lon",
    aimms_cloud_col: Optional[str] = None,
    fims_time_col: Optional[str] = None,
    fims_cloud_col: str = "Cloud_flag",
    fims_density_measure: str = "ln",   # "ln" | "log10" | "per_bin"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (Dp_lo_nm, Dp_hi_nm, avg_N_cm3_per_bin, std_N_cm3_per_bin).

    IMPORTANT:
      - If FIMS reports density (dN/dlnDp or dN/dlog10Dp), we convert to per-bin N using bin widths.
      - If FIMS already reports per-bin N, set fims_density_measure="per_bin".
    """
    aimms = _read_aimms(aimms_file)
    fims = _read_fims_core(fims_file)

    # determine FIMS time col
    if fims_time_col is None:
        for cand in ("Time(UTC)", "UTC", "Start_UTC"):
            if cand in fims:
                fims_time_col = cand
                break
    if fims_time_col is None:
        raise KeyError("Could not identify FIMS time column; set config['fims_time_col'].")

    fims_idx = _fims_time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        fims=fims,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        fims_time_col=fims_time_col,
        fims_cloud_col=fims_cloud_col,
    )
    if fims_idx.size == 0:
        raise RuntimeError("No matching FIMS indices found for requested altitude/region/cloud filters.")

    Nbins = np.asarray(fims["_N_bins"][fims_idx, :], dtype=float)  # (nt, nbins)
    avg_density = np.nanmean(Nbins, axis=0)
    std_density = np.nanstd(Nbins, axis=0)

    nbins = avg_density.size
    Dp_lo_nm, Dp_hi_nm = _read_fims_bins_file(fims_bins_file, expected_bins=nbins)

    # Apply max_dp filter
    keep = np.where(Dp_hi_nm <= float(max_dp_nm))[0]
    if keep.size == 0:
        raise RuntimeError(f"max_dp_nm={max_dp_nm} removed all FIMS bins.")
    Dp_lo_nm = Dp_lo_nm[keep]
    Dp_hi_nm = Dp_hi_nm[keep]
    avg_density = avg_density[keep]
    std_density = std_density[keep]

    # Convert density -> per-bin N if needed
    measure = (fims_density_measure or "ln").strip().lower()
    if measure == "per_bin":
        N_bin = avg_density
        N_bin_std = std_density
    else:
        if np.any(Dp_lo_nm <= 0) or np.any(Dp_hi_nm <= 0) or np.any(Dp_hi_nm <= Dp_lo_nm):
            raise ValueError("Invalid bin edges in bins file (need positive, hi>lo).")
        dln = np.log(Dp_hi_nm / Dp_lo_nm)
        if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
            raise ValueError("Invalid log bin widths computed from bin edges.")

        if measure in ("ln", "dndln", "dndlnd"):
            N_bin = avg_density * dln
            N_bin_std = std_density * dln
        elif measure in ("log10", "dndlog10"):
            dlog10 = dln / math.log(10.0)
            N_bin = avg_density * dlog10
            N_bin_std = std_density * dlog10
        else:
            raise ValueError(f"Unknown fims_density_measure '{fims_density_measure}'")

    return Dp_lo_nm, Dp_hi_nm, N_bin, N_bin_std


# -----------------------------
# miniSPLAT number fractions
# -----------------------------

def _read_minisplat_number_fractions(
    *,
    splat_file: str,
    aimms_file: str,
    fims_file: str,
    splat_species: Dict[str, List[str]],
    z: float,
    dz: float,
    cloud_flag_value: Optional[float] = 0,
    region_filter: Optional[Dict[str, float]] = None,
    aimms_time_col: str = "Time(UTC)",
    aimms_alt_col: str = "Alt",
    aimms_lat_col: str = "Lat",
    aimms_lon_col: str = "Lon",
    aimms_cloud_col: Optional[str] = None,
    fims_time_col: Optional[str] = None,
    fims_cloud_col: str = "Cloud_flag",
    splat_time_col: str = "Time",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (avg_comp, comp_err) for reduced classes.
    Uses the same filtered FIMS times as the size distribution selection.
    """
    aimms = _read_aimms(aimms_file)
    fims = _read_fims_core(fims_file)
    splat = _read_delimited_table_with_header(splat_file)

    # resolve columns
    if fims_time_col is None:
        for cand in ("Time(UTC)", "UTC", "Start_UTC"):
            if cand in fims:
                fims_time_col = cand
                break
    if fims_time_col is None:
        raise KeyError("Could not identify FIMS time column for miniSPLAT matching; set config['fims_time_col'].")

    if splat_time_col not in splat:
        if "Time(UTC)" in splat:
            splat_time_col = "Time(UTC)"
        else:
            raise KeyError(
                f"miniSPLAT missing time column '{splat_time_col}'. Available: {sorted(splat.keys())[:50]}"
            )

    fims_idx = _fims_time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        fims=fims,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        fims_time_col=fims_time_col,
        fims_cloud_col=fims_cloud_col,
    )
    if fims_idx.size == 0:
        raise RuntimeError("No FIMS indices for miniSPLAT matching (alt/region/cloud filters too strict).")

    fims_times = np.unique(np.asarray(fims[fims_time_col], dtype=float)[fims_idx])
    t_splat = np.asarray(splat[splat_time_col], dtype=float)

    splat_indices: List[int] = []
    for t in fims_times:
        rows = np.where(t_splat == t)[0]
        if rows.size:
            splat_indices.extend(rows.tolist())
    splat_indices = np.unique(np.array(splat_indices, dtype=int))
    if splat_indices.size == 0:
        raise RuntimeError("No miniSPLAT rows matched the filtered FIMS times.")

    avg_comp: Dict[str, float] = {}
    comp_err: Dict[str, float] = {}

    for reduced, cols in splat_species.items():
        summation = np.zeros(splat_indices.size, dtype="float64")
        for c in cols:
            if c not in splat:
                raise KeyError(
                    f"miniSPLAT missing column '{c}' referenced by splat_species['{reduced}']. "
                    f"Available (sample): {sorted(splat.keys())[:50]}"
                )
            summation += np.asarray(splat[c], dtype=float)[splat_indices]
        avg_comp[reduced] = float(np.nanmean(summation))
        comp_err[reduced] = float(np.nanstd(summation))

    # normalize defensively
    s = sum(avg_comp.values())
    if s > 0:
        for k in list(avg_comp.keys()):
            avg_comp[k] /= s
            comp_err[k] /= s

    return avg_comp, comp_err


# -----------------------------
# AMS mass fractions
# -----------------------------

def _read_ams_mass_fractions(
    *,
    ams_file: str,
    aimms_file: str,
    fims_file: str,
    z: float,
    dz: float,
    cloud_flag_value: Optional[float] = 0,
    region_filter: Optional[Dict[str, float]] = None,
    aimms_time_col: str = "Time(UTC)",
    aimms_alt_col: str = "Alt",
    aimms_lat_col: str = "Lat",
    aimms_lon_col: str = "Lon",
    aimms_cloud_col: Optional[str] = None,
    fims_time_col: Optional[str] = None,
    fims_cloud_col: str = "Cloud_flag",
    ams_time_col: Optional[str] = None,
    ams_flag_col: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Returns:
      (mass_frac, mass_frac_err, measured_mass_mean, measured_mass_std)

    mass_frac keys: SO4, NO3, OC, NH4
    """
    aimms = _read_aimms(aimms_file)
    fims = _read_fims_core(fims_file)
    ams = _read_icartt_table(ams_file)

    if fims_time_col is None:
        for cand in ("Time(UTC)", "UTC", "Start_UTC"):
            if cand in fims:
                fims_time_col = cand
                break
    if fims_time_col is None:
        raise KeyError("Could not identify FIMS time column for AMS matching; set config['fims_time_col'].")

    # FIMS indices for alt/region (and cloud if FIMS has it)
    fims_idx = _fims_time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        fims=fims,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        fims_time_col=fims_time_col,
        fims_cloud_col=fims_cloud_col,
    )
    if fims_idx.size == 0:
        raise RuntimeError("No FIMS indices for AMS matching (alt/region/cloud filters too strict).")

    fims_times = np.unique(np.asarray(fims[fims_time_col], dtype=float)[fims_idx])

    # AMS time + flag columns
    ams_cols = list(ams.keys())

    if ams_time_col is None:
        for cand in ("dat_ams_utc", "Time", "Time(UTC)", "Start_UTC", "UTC", "time"):
            if cand in ams:
                ams_time_col = cand
                break
    if ams_time_col is None:
        raise KeyError(
            f"Could not identify AMS time column. Set config['ams_time_col']. "
            f"Available (sample): {sorted(ams_cols)[:50]}"
        )

    if ams_flag_col is None:
        for cand in ("flag", "Flag", "qc_flag", "QC_Flag", "qc", "QC"):
            if cand in ams:
                ams_flag_col = cand
                break
    if ams_flag_col is None:
        raise KeyError(
            f"Could not identify AMS flag/QC column. Set config['ams_flag_col']. "
            f"Available (sample): {sorted(ams_cols)[:50]}"
        )

    t_ams = np.asarray(ams[ams_time_col], dtype=float)
    flag = np.asarray(ams[ams_flag_col], dtype=float)

    ams_idx: List[int] = []
    for t in fims_times:
        rows = np.where((t_ams == t) & (flag < 0.5))[0]
        if rows.size:
            ams_idx.extend(rows.tolist())
    ams_idx = np.unique(np.array(ams_idx, dtype=int))
    if ams_idx.size == 0:
        raise RuntimeError("No AMS rows matched filtered FIMS times with flag < 0.5.")

    # Required AMS mass columns
    need = ["Org", "NO3", "SO4", "NH4"]
    missing = [k for k in need if k not in ams]
    if missing:
        raise KeyError(
            f"AMS missing required mass columns {missing}. Available (sample): {sorted(ams_cols)[:60]}"
        )

    org = np.asarray(ams["Org"], dtype=float)[ams_idx]
    no3 = np.asarray(ams["NO3"], dtype=float)[ams_idx]
    so4 = np.asarray(ams["SO4"], dtype=float)[ams_idx]
    nh4 = np.asarray(ams["NH4"], dtype=float)[ams_idx]

    total = org + no3 + so4 + nh4
    ok = np.where(np.isfinite(total) & (total > 0))[0]
    if ok.size == 0:
        raise RuntimeError("AMS total mass is nonpositive or nonfinite for all matched points.")
    org = org[ok]; no3 = no3[ok]; so4 = so4[ok]; nh4 = nh4[ok]; total = total[ok]

    frac_org = org / total
    frac_no3 = no3 / total
    frac_so4 = so4 / total
    frac_nh4 = nh4 / total

    mass_frac = {
        "OC": float(np.nanmean(frac_org)),
        "NO3": float(np.nanmean(frac_no3)),
        "SO4": float(np.nanmean(frac_so4)),
        "NH4": float(np.nanmean(frac_nh4)),
    }
    mass_frac_err = {
        "OC": float(np.nanstd(frac_org)),
        "NO3": float(np.nanstd(frac_no3)),
        "SO4": float(np.nanstd(frac_so4)),
        "NH4": float(np.nanstd(frac_nh4)),
    }

    # normalize defensively
    s = sum(mass_frac.values())
    if s > 0:
        for k in list(mass_frac.keys()):
            mass_frac[k] /= s
            mass_frac_err[k] /= s

    measured_mass = float(np.nanmean(total))
    measured_mass_err = float(np.nanstd(total))
    return mass_frac, mass_frac_err, measured_mass, measured_mass_err


# -----------------------------
# Composition helpers
# -----------------------------

def _normalize_fracs(fracs: Dict[str, float]) -> Dict[str, float]:
    fracs2 = {k: float(v) for k, v in fracs.items() if np.isfinite(v) and float(v) > 0.0}
    s = sum(fracs2.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in fracs2.items()}


def _default_template_for_type(
    ptype: str,
    ams_mass_frac: Dict[str, float],
    ams_species_map: Dict[str, str],
) -> Dict[str, float]:
    p = ptype.strip().upper()
    if p == "BC":
        return {"BC": 1.0}
    if p == "OIN":
        return {"OIN": 1.0}

    tmpl: Dict[str, float] = {}
    for ams_k, v in ams_mass_frac.items():
        spec = ams_species_map.get(ams_k, ams_k)
        tmpl[spec] = float(v)
    return _normalize_fracs(tmpl)


def _composition_for_type(
    ptype: str,
    strategy: str,
    type_templates: Optional[Dict[str, Dict[str, float]]],
    ams_mass_frac: Dict[str, float],
    ams_species_map: Dict[str, str],
) -> Dict[str, float]:
    strategy2 = (strategy or "ams_everywhere").strip().lower()
    templates = type_templates or {}

    if strategy2 == "templates_only":
        if ptype not in templates:
            raise KeyError(f"composition_strategy=templates_only but missing template for type '{ptype}'")
        return _normalize_fracs(templates[ptype])

    if strategy2 == "templates_fallback_to_ams":
        if ptype in templates:
            return _normalize_fracs(templates[ptype])
        return _default_template_for_type(ptype, ams_mass_frac, ams_species_map)

    # default
    return _default_template_for_type(ptype, ams_mass_frac, ams_species_map)


# -----------------------------
# Builder entrypoint
# -----------------------------

@register("hiscale_observations")
def build(config: Dict[str, Any]) -> ParticlePopulation:
    config = normalize_population_config(config)

    required = ["fims_file", "fims_bins_file", "aimms_file", "splat_file", "ams_file", "z", "dz", "splat_species"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"hiscale_observations missing required config keys: {missing}")

    fims_file = str(config["fims_file"])
    fims_bins_file = str(config["fims_bins_file"])
    aimms_file = str(config["aimms_file"])
    splat_file = str(config["splat_file"])
    ams_file = str(config["ams_file"])

    z = float(config["z"])
    dz = float(config["dz"])
    splat_species = dict(config["splat_species"])

    cloud_flag_value = config.get("cloud_flag_value", 0)
    max_dp_nm = float(config.get("max_dp_nm", 1000.0))
    splat_cutoff_nm = float(config.get("splat_cutoff_nm", 0.0))
    region_filter = config.get("region_filter", None)

    composition_strategy = str(config.get("composition_strategy", "ams_everywhere"))
    type_templates = config.get("type_templates", None)
    ams_species_map = dict(config.get("ams_species_map", {"SO4": "SO4", "NO3": "NO3", "OC": "OC", "NH4": "NH4"}))

    species_modifications = config.get("species_modifications", {})
    D_is_wet = bool(config.get("D_is_wet", False))

    aimms_time_col = str(config.get("aimms_time_col", "Time(UTC)"))
    aimms_alt_col = str(config.get("aimms_alt_col", "Alt"))
    aimms_lat_col = str(config.get("aimms_lat_col", "Lat"))
    aimms_lon_col = str(config.get("aimms_lon_col", "Lon"))
    aimms_cloud_col = config.get("aimms_cloud_col", None)  # usually None

    fims_time_col = config.get("fims_time_col", None)      # usually None; auto-detect
    fims_cloud_col = str(config.get("fims_cloud_col", "Cloud_flag"))

    fims_density_measure = str(config.get("fims_density_measure", "ln"))

    # --- read FIMS average distribution ---
    Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3 = _read_fims_avg_size_dist(
        fims_file=fims_file,
        fims_bins_file=fims_bins_file,
        aimms_file=aimms_file,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        max_dp_nm=max_dp_nm,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        fims_time_col=fims_time_col,
        fims_cloud_col=fims_cloud_col,
        fims_density_measure=fims_density_measure,
    )
    Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)

    # --- miniSPLAT reduced-class number fractions ---
    type_fracs, type_fracs_err = _read_minisplat_number_fractions(
        splat_file=splat_file,
        aimms_file=aimms_file,
        fims_file=fims_file,
        splat_species=splat_species,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        fims_time_col=fims_time_col,
        fims_cloud_col=fims_cloud_col,
    )
    tf = _normalize_fracs(type_fracs)
    if not tf:
        raise RuntimeError("miniSPLAT type fractions sum to 0; cannot build population.")

    # --- AMS mass fractions + measured total mass (for optional scaling) ---
    ams_mass_frac, ams_mass_frac_err, measured_mass, measured_mass_err = _read_ams_mass_fractions(
        ams_file=ams_file,
        aimms_file=aimms_file,
        fims_file=fims_file,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        fims_time_col=fims_time_col,
        fims_cloud_col=fims_cloud_col,
        ams_time_col=config.get("ams_time_col", None),
        ams_flag_col=config.get("ams_flag_col", None),
    )

    # --- determine population species ---
    pop_species_names: List[str] = []

    if isinstance(type_templates, dict):
        for tmpl in type_templates.values():
            if isinstance(tmpl, dict):
                for s in tmpl.keys():
                    if s not in pop_species_names:
                        pop_species_names.append(s)

    for ams_k in ams_mass_frac.keys():
        mapped = ams_species_map.get(ams_k, ams_k)
        if mapped not in pop_species_names:
            pop_species_names.append(mapped)

    for ptype in tf.keys():
        up = ptype.strip().upper()
        if up in ("BC", "OIN") and up not in pop_species_names:
            pop_species_names.append(up)

    pop_species_list = tuple(
        get_species(spec_name, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    pop = ParticlePopulation(
        species=pop_species_list,
        spec_masses=[],
        num_concs=[],
        ids=[],
        species_modifications=species_modifications,
    )

    # --- units ---
    # N_cm3 is per-bin number concentration (cm^-3) returned by _read_fims_avg_size_dist.
    # Convert to m^-3 for particle allocation and derive dN/dlnD for diagnostics/metadata.
    Dp_mid_m = Dp_mid_nm * 1e-9
    N_m3 = N_cm3 * 1e6  # cm^-3 -> m^-3

    dln = np.log(Dp_hi_nm / Dp_lo_nm)
    if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
        raise ValueError("Invalid FIMS bin edges; cannot compute dln widths.")
    dNdln_m3 = N_m3 / dln

    part_id = 0
    for i in range(len(Dp_mid_m)):
        if not np.isfinite(N_m3[i]) or N_m3[i] <= 0:
            continue

        # separate_tools behavior: below splat_cutoff_nm -> force AS
        if splat_cutoff_nm > 0 and (Dp_mid_nm[i] < splat_cutoff_nm):
            tf_local = {"AS": 1.0}
        else:
            tf_local = tf

        for ptype, frac in tf_local.items():
            Ni_t = float(N_m3[i] * frac)
            if not np.isfinite(Ni_t) or Ni_t <= 0:
                continue

            fracs_dict = _composition_for_type(
                ptype=ptype,
                strategy=composition_strategy,
                type_templates=type_templates,
                ams_mass_frac=ams_mass_frac,
                ams_species_map=ams_species_map,
            )
            fracs_dict = _normalize_fracs(fracs_dict)
            if not fracs_dict:
                continue

            spec_to_frac = dict(fracs_dict)
            pop_aligned_fracs = [spec_to_frac.get(name, 0.0) for name in pop_species_names]

            particle = make_particle(
                Dp_mid_m[i],
                pop_species_list,
                pop_aligned_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet,
            )

            part_id += 1
            pop.set_particle(particle, part_id, Ni_t, suppress_warning=True)

    if len(pop.ids) == 0:
        raise RuntimeError("Built zero particles; check filters and input files.")

    # --- optional AMS mass closure scaling (same style as separate_tools) ---
    try:
        spec_masses = np.asarray(pop.spec_masses, dtype=float)   # (npart, nspec) kg/particle
        num_concs = np.asarray(pop.num_concs, dtype=float)       # (npart,) m^-3
        if spec_masses.ndim == 2 and spec_masses.shape[0] == num_concs.shape[0]:
            total_mass_kg_m3 = float(np.sum(num_concs * np.sum(spec_masses, axis=1)))
            if np.isfinite(total_mass_kg_m3) and total_mass_kg_m3 > 0 and np.isfinite(measured_mass) and measured_mass > 0:
                # measured_mass is assumed ug/m3; convert kg/m3 -> ug/m3 via 1e9
                scale_factor = float(measured_mass) / (1.0e9 * total_mass_kg_m3)
                pop.num_concs = list(num_concs * scale_factor)
    except Exception:
        # Keep builder robust; scaling is optional
        pass

    # Optional metadata
    try:
        pop.metadata = {
            "source": "hiscale_observations",
            "z": z,
            "dz": dz,
            "cloud_flag_value": cloud_flag_value,
            "max_dp_nm": max_dp_nm,
            "splat_cutoff_nm": splat_cutoff_nm,
            "fims_density_measure": fims_density_measure,
            "type_fracs": tf,
            "type_fracs_err": type_fracs_err,
            "ams_mass_frac": ams_mass_frac,
            "ams_mass_frac_err": ams_mass_frac_err,
            "measured_ams_total_mass": measured_mass,
            "measured_ams_total_mass_err": measured_mass_err,
            "fims_size_distribution": {
                "Dp_lo_nm": Dp_lo_nm.copy(),
                "Dp_hi_nm": Dp_hi_nm.copy(),
                "dln": dln.copy(),
                "N_bin_m3": N_m3.copy(),
                "dNdlnD_m3": dNdln_m3.copy(),
            },
        }
    except Exception:
        pass

    # Sanity check: total N (note scaling may change it)
    try:
        N_expected = float(np.nansum(N_m3[np.isfinite(N_m3) & (N_m3 > 0)]))
        N_built = float(np.nansum(np.asarray(pop.num_concs, dtype=float)))
        if N_expected > 0 and np.isfinite(N_built):
            rel = abs(N_built - N_expected) / N_expected
            if rel > 1e-3:
                print(f"WARNING[hiscale_observations]: total N mismatch expected={N_expected:.6e}, built={N_built:.6e}")
    except Exception:
        pass

    return pop
