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
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.special import erf
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import numpy as np

import matplotlib.pyplot as plt
from ..utils import normalize_population_config
from part2pop import build_population
from pathlib import Path
from .registry import register
from part2pop.species.registry import retrieve_one_species

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

# -----------------------------
# BEASD core + bins
# -----------------------------

def _read_beasd_core(beasd_file: str) -> Dict[str, Any]:
    """
    Parse BEAST via ICARTT header + numeric columns.
    Detect size-bin columns by header pattern (C_*). Returns:
      - all columns
      - "_bin_cols": ordered list of bin col names
      - "_N_bins": (nt, nbins) array
    """
    table = _read_icartt_table(beasd_file)
    keys = list(table.keys())

    # Common FIMS bin col patterns: C_x, C_y, ...
    bin_cols = [k for k in keys if re.match(r"(?i)^C[_\-]", k)]

    # If that fails, try selecting a contiguous block after the first column (time)
    if not bin_cols and len(keys) >= 8:
        # choose the longest block starting at index 1
        # (ICARTT already gave numeric arrays for all cols, so this is mostly about intent)
        best_start = 1
        best_len = len(keys) - 9
        # require many bins to accept
        if best_len >= 10:
            bin_cols = keys[best_start:best_start + best_len]

    if not bin_cols:
        raise ValueError(
            f"Could not identify BEASD bin columns in {beasd_file}. "
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

def _read_beasd_bins(beasd_file: str, expected_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse the BEASD file and return (Dp_lowers_nm, Dp_uppers_nm) with length expected_bins.
    Looks for "diameter range from" in lines of file that define bin edges.
    """
    lines = _read_lines(beasd_file)
    idx = np.array([i for i, line in enumerate(lines) if "diameter range from" in line], dtype=int)
    if idx.size == expected_bins:
        lo_list: List[float] = []
        hi_list: List[float] = []
        for ii in idx:
            jj = np.where(np.array(lines[ii].split()) == "from")[0][0]
            lo_list.append(float(lines[ii].split()[jj+1]))
            jj = np.where(np.array(lines[ii].split()) == "to")[0][0]
            hi_list.append(float(lines[ii].split()[jj+1]))
            if len(lo_list) == expected_bins and len(hi_list) == expected_bins:
                lo = np.array(lo_list, dtype="float64")
                hi = np.array(hi_list, dtype="float64")
                if np.all(hi > lo) and np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0):
                    return lo, hi
    else:
        raise ValueError(f"Could not find expected number of bin lines in BEASD file {beasd_file}. Found {idx.size}, expected {expected_bins}.")

# -----------------------------
# Time / altitude / region / cloud selection (works for BEASD and FIMS)
# -----------------------------

def _time_indices_for_altitude_and_cloudflag(
    *,
    size_dist: Dict[str, Any],
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
    size_dist_time_col: str = "Time(UTC)",
    size_dist_cloud_col: str = "Cloud_flag",
) -> np.ndarray:
    """
    Returns indices into FIMS or BEASD rows that correspond to AIMMS points in the requested
    altitude window (+ region) by matching rounded AIMMS times to FIMS/BEASD times.

    Cloud handling:
      - AIMMS: only applied if aimms_cloud_col is provided AND exists.
      - FIMS/BEASD: only applied if size_dist_cloud_col exists in size_dist.
        If it doesn't exist, we do NOT error; we simply skip that filter.
    """
    if region_filter is None:
        region_filter = {"lon_min": -97.5, "lon_max": -97.4, "lat_min": 36.05, "lat_max": 36.81}

    for k in (aimms_time_col, aimms_alt_col, aimms_lat_col, aimms_lon_col):
        if k not in aimms:
            raise KeyError(f"AIMMS missing required column '{k}'. Available: {sorted(aimms.keys())[:50]}")

    if size_dist_time_col not in size_dist:
        raise KeyError(f"FIMS/BEASD missing time column '{size_dist_time_col}'. Available: {sorted(size_dist.keys())[:50]}")

    t_aimms = aimms[aimms_time_col]
    alt = aimms[aimms_alt_col]
    lat = aimms[aimms_lat_col]
    lon = aimms[aimms_lon_col]

    in_region = (
        (lon > region_filter["lon_min"]) & (lon < region_filter["lon_max"]) &
        (lat > region_filter["lat_min"]) & (lat < region_filter["lat_max"])
    )

    idx_alt = np.where((alt >= z - dz) & (alt < z + dz) & (in_region))[0]
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
    t_size_dist = np.asarray(size_dist[size_dist_time_col], dtype=float)    
    mask = np.isin(t_size_dist, aimms_times)

    # optional FIMS/BEASD cloud-flag filter (ONLY if the column exists)
    if cloud_flag_value is not None and (size_dist_cloud_col in size_dist):
        mask = mask & (np.asarray(size_dist[size_dist_cloud_col], dtype=float) == float(cloud_flag_value))

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

    fims_idx = _time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        size_dist=fims,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=fims_time_col,
        size_dist_cloud_col=fims_cloud_col,
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
# BEASD average size distribution
# -----------------------------

def _read_beasd_avg_size_dist(
    *,
    beasd_file: str,
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
    beasd_time_col: Optional[str] = None,
    beasd_cloud_col: str = "Cloud_flag",
    beasd_density_measure: str = "per_bin",   # "ln" | "log10" | "per_bin"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (Dp_lo_nm, Dp_hi_nm, avg_N_cm3_per_bin, std_N_cm3_per_bin).

    IMPORTANT:
      - If BEASD reports density (dN/dlnDp or dN/dlog10Dp), we convert to per-bin N using bin widths.
      - If BEASD already reports per-bin N, set beasd_density_measure="per_bin".
    """
    aimms = _read_aimms(aimms_file)
    beasd = _read_beasd_core(beasd_file)

    # determine BEASD time col
    if beasd_time_col is None:
        for cand in ("Time(UTC)", "UTC", "Start_UTC"):
            if cand in beasd:
                beasd_time_col = cand
                break
    if beasd_time_col is None:
        raise KeyError("Could not identify BEASD time column; set config['beasd_time_col'].")

    beasd_idx = _time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        size_dist=beasd,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=beasd_time_col,
        size_dist_cloud_col=beasd_cloud_col,
    )

    if beasd_idx.size == 0:
        raise RuntimeError("No matching BEASD indices found for requested altitude/region/cloud filters.")

    Nbins = np.asarray(beasd["_N_bins"][beasd_idx, :], dtype=float)  # (nt, nbins)
    avg_density = np.nanmean(Nbins, axis=0)
    std_density = np.nanstd(Nbins, axis=0)

    nbins = avg_density.size
    Dp_lo_nm, Dp_hi_nm = _read_beasd_bins(beasd_file, expected_bins=nbins)

    # Apply max_dp filter
    keep = np.where(Dp_hi_nm <= float(max_dp_nm))[0]
    if keep.size == 0:
        raise RuntimeError(f"max_dp_nm={max_dp_nm} removed all FIMS bins.")
    Dp_lo_nm = Dp_lo_nm[keep]
    Dp_hi_nm = Dp_hi_nm[keep]
    avg_density = avg_density[keep]
    std_density = std_density[keep]

    # Convert density -> per-bin N if needed
    measure = (beasd_density_measure or "ln").strip().lower()
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
            raise ValueError(f"Unknown beasd_density_measure '{fims_density_measure}'")

    return Dp_lo_nm, Dp_hi_nm, N_bin, N_bin_std

# -----------------------------
# miniSPLAT number fractions with FIMS size distribtion
# -----------------------------

def _read_minisplat_number_fractions(
    *,
    splat_file: str,
    aimms_file: str,
    size_dist_type: str,
    size_dist_file: str,
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
    size_dist_time_col: Optional[str] = None,
    size_dist_cloud_col: str = "Cloud_flag",
    splat_time_col: str = "Time",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (avg_comp, comp_err) for reduced classes.
    Uses the same filtered FIMS times as the size distribution selection.
    """    
    aimms = _read_aimms(aimms_file)
    
    if size_dist_type == "FIMS":
        size_dist = _read_fims_core(size_dist_file)
        
        # resolve columns
        if size_dist_time_col is None:
            for cand in ("Time(UTC)", "UTC", "Start_UTC"):
                if cand in size_dist:
                    size_dist_time_col = cand
                    break
        if size_dist_time_col is None:
            raise KeyError("Could not identify FIMS time column for miniSPLAT matching; set config['fims_time_col'].")
    
    elif size_dist_type == "BEASD":
        size_dist = _read_beasd_core(size_dist_file)
        
        # resolve columns
        if size_dist_time_col is None:
            for cand in ("Time(UTC)", "UTC", "Start_UTC"):
                if cand in size_dist:
                    size_dist_time_col = cand
                    break
        if size_dist_time_col is None:
            raise KeyError("Could not identify BEASD time column for miniSPLAT matching; set config['beasd_time_col'].")

    else:
        raise ValueError("Either fims_file or beasd_file must be provided for miniSPLAT matching.")

    splat = _read_delimited_table_with_header(splat_file)
    if splat_time_col not in splat:
        if "Time(UTC)" in splat:
            splat_time_col = "Time(UTC)"
        else:
            raise KeyError(
                f"miniSPLAT missing time column '{splat_time_col}'. Available: {sorted(splat.keys())[:50]}"
            )

    size_dist_idx = _time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        size_dist=size_dist,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=size_dist_time_col,
        size_dist_cloud_col=size_dist_cloud_col,
    )
    if size_dist_idx.size == 0:
        raise RuntimeError("No FIMS/BEASD indices for miniSPLAT matching (alt/region/cloud filters too strict).")

    size_dist_times = np.unique(np.asarray(size_dist[size_dist_time_col], dtype=float)[size_dist_idx])
    t_splat = np.asarray(splat[splat_time_col], dtype=float)

    splat_indices: List[int] = []
    for t in size_dist_times:
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
# AMS mass fractions for FIMS size distribution
# -----------------------------

def _read_ams_mass_fractions(
    *,
    ams_file: str,
    aimms_file: str,
    size_dist_type: str,
    size_dist_file: str,
    z: float,
    dz: float,
    cloud_flag_value: Optional[float] = 0,
    region_filter: Optional[Dict[str, float]] = None,
    aimms_time_col: str = "Time(UTC)",
    aimms_alt_col: str = "Alt",
    aimms_lat_col: str = "Lat",
    aimms_lon_col: str = "Lon",
    aimms_cloud_col: Optional[str] = None,
    size_dist_time_col: Optional[str] = None,
    size_dist_cloud_col: str = "Cloud_flag",
    ams_time_col: Optional[str] = None,
    ams_flag_col: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Returns:
      (mass_frac, mass_frac_err, measured_mass_mean, measured_mass_std)

    mass_frac keys: SO4, NO3, OC, NH4
    """
    aimms = _read_aimms(aimms_file)
    
    if size_dist_type == "FIMS":
        size_dist = _read_fims_core(size_dist_file)
        
        # resolve columns
        if size_dist_time_col is None:
            for cand in ("Time(UTC)", "UTC", "Start_UTC"):
                if cand in size_dist:
                    size_dist_time_col = cand
                    break
        if size_dist_time_col is None:
            raise KeyError("Could not identify FIMS time column for miniSPLAT matching; set config['fims_time_col'].")
    
    elif size_dist_type == "BEASD":
        size_dist = _read_beasd_core(size_dist_file)
        
        # resolve columns
        if size_dist_time_col is None:
            for cand in ("Time(UTC)", "UTC", "Start_UTC"):
                if cand in size_dist:
                    size_dist_time_col = cand
                    break
        if size_dist_time_col is None:
            raise KeyError("Could not identify BEASD time column for miniSPLAT matching; set config['beasd_time_col'].")

    else:
        raise ValueError("Either fims_file or beasd_file must be provided for miniSPLAT matching.")
        
    ams = _read_icartt_table(ams_file)

    # FIMS/BEASD indices for alt/region (and cloud if FIMS/BEASD has it)
    size_dist_idx = _time_indices_for_altitude_and_cloudflag(
        aimms=aimms,
        size_dist=size_dist,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=size_dist_time_col,
        size_dist_cloud_col=size_dist_cloud_col,
    )
    if size_dist_idx.size == 0:
        raise RuntimeError("No FIMS/BEASD indices for AMS matching (alt/region/cloud filters too strict).")

    size_dist_times = np.unique(np.asarray(size_dist[size_dist_time_col], dtype=float)[size_dist_idx])

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
    for t in size_dist_times:
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

def lognormal_distribution(x, Ntot, Dpg, sigma):
    """Returns N for lognormal distribution with total number Ntot, geometric mean Dpg, and geometric stddev sigma, evaluated at x."""
    prefactor = Ntot/(np.sqrt(2.0*np.pi)*np.log(sigma)*x)
    numerator = -1.0*np.power(np.log(x)-np.log(Dpg), 2)
    denominator = 2.0*np.log(sigma)*np.log(sigma)
    N = prefactor*np.exp(numerator/denominator)
    return N

def Nmodal_lognormal(x, *params):
    """Returns linear composition of N lognormal distributions."""
    n_modes=int(len(params)/3)
    N=np.zeros(len(x))
    for mode in range(n_modes):
        Ntot=params[mode*3+0]
        Dpg=params[mode*3+1]
        sigma=params[mode*3+2]
        N=N+lognormal_distribution(x, Ntot, Dpg, sigma)
    return N

def fit_Nmodal_distibution(
    Dp: np.ndarray, 
    N: np.ndarray):
    modes=0
    r2=0
    while r2<0.9:
        modes+=1
        try:
            p0 = []
            lower_bounds = []
            upper_bounds = []
            percentiles=np.linspace(0, 100, modes+2)[1:-1]
            for ii in range(modes):
                p0.append(trapezoid(N, Dp)/modes)
                p0.append(10**(np.percentile(np.log10(Dp), percentiles[ii])))
                p0.append(1.3)
                lower_bounds.append(0)
                lower_bounds.append(0)
                lower_bounds.append(1.0)
                upper_bounds.append(np.inf)
                upper_bounds.append(np.inf)
                upper_bounds.append(np.inf)
            pars, cov = curve_fit(Nmodal_lognormal, xdata=Dp, ydata=N, p0=p0, bounds=[lower_bounds, upper_bounds])
            fitted_N = Nmodal_lognormal(Dp, *pars)       
            rss = np.sum((N - fitted_N) ** 2)
            tss = np.sum((N - np.mean(N)) ** 2)
            r2 = 1 - (rss / tss)
        except:
            r2 = 0  
    modes = int(len(pars)/3)
    output_pars=[]
    for mode in range(modes):
        output_pars.append(pars[mode*3:mode*3+3])          
    return output_pars

def size_dependent_composition(
    Dps: np.ndarray, 
    measured_Ns: np.ndarray, 
    N_modes: int, 
    splat_cutoff_nm: float, 
    splat_number_fraction: float, 
    mode_fractions: np.ndarray, 
    size_dist_params: List[float]):
    
    CDF_calc=np.zeros(len(Dps))
    for mode in range(N_modes):
        Dpg=size_dist_params[mode*3+1]
        sigma=size_dist_params[mode*3+2]
        CDF_calc+= mode_fractions[mode]*0.5*(1+erf((np.log(Dps)-np.log(Dpg))/(np.sqrt(2)*np.log(sigma))))

    indices = np.where(1e9*Dps >= splat_cutoff_nm)
    Ntot_meas_gt_cutoff = np.sum(measured_Ns[indices[0][0]:])
    Fx_gt_cutoff = 1 - CDF_calc[indices[0][0]]
    Nx_gt_cutoff = Ntot_meas_gt_cutoff*splat_number_fraction
    Nx = Nx_gt_cutoff/Fx_gt_cutoff
    
    for mode in range(N_modes):
        size_dist_params[mode*3+0]=Nx*mode_fractions[mode]
    
    Ns=Nmodal_lognormal(Dps, *size_dist_params)
    mult = (Nx_gt_cutoff)/np.sum(Ns[indices[0][0]:])
    Ns*=mult
    
    return Ns

def optimize_splat_species_distributions(
    *,
    splat_species: Dict[str, List[str]],
    size_distribution_pars: List[float],
    datapoints: int = 1000,
    measured_Dp: np.ndarray, 
    measured_N: np.ndarray,
    splat_number_fractions: Dict[str, float],
    ams_mass_fractions: Dict[str, float],
    splat_cutoff_nm: float = 85.0,
    mass_thresholds: Dict[str, Tuple[float, str]]):

    Dpg_BC = 110e-9
    sigma_BC = 1.6
    Dpg_dust = 110e-9 # taken from accumulation mode of MAM4
    sigma_dust = 1.6
    
    # get species that need optimizing (i.e. not BC or OIN)
    model_species=[]
    for spec in splat_species.keys():
        if spec!='BC' and spec!='OIN':
            model_species.append(spec)

    # Draw random fractions for each particle in each mode
    modes=len(size_distribution_pars)
    Nspec=len(model_species)
    mode_fractions = np.random.rand(datapoints, Nspec, modes)
    mode_fractions /= mode_fractions.sum(axis=2, keepdims=True)
    mode_fractions = mode_fractions.reshape(datapoints, Nspec * modes)
    # mode_fractions = np.unique(mode_fractions, axis=0)

    # get initial size distribution properties from measurements
    measured_Ntot = np.sum(measured_N) # N per volume
    measured_mean_size = np.average(measured_Dp, weights=measured_N)
    SA_dist = 4.0*np.pi*np.power((measured_Dp)/2, 2)*(measured_N) # aerosol SA/volume air
    measured_SA = np.sum(SA_dist) # total surface area per volume
    V_dist = (4.0/3.0)*np.pi*np.power((measured_Dp)/2, 3)*measured_N # volume aerosol/volume air
    measured_Vtot = np.sum(V_dist) # total surface area per volume
    
    min_RSS = 1e10
    line_to_save=-1
    size_distribution_pars=list(np.array(size_distribution_pars).flatten())
    for line in range(len(mode_fractions)):
        
        spec_masses={}
        total_mass=0
        total_Ns=np.zeros(len(measured_Dp))
        
        for spec in range(Nspec):
            params=[]
            for mode in range(modes):
                params.append(splat_number_fractions[model_species[spec]]*mode_fractions[line, spec*modes+mode])
                params.append(size_distribution_pars[mode*3+1])
                params.append(size_distribution_pars[mode*3+2])

            spec_Ns = size_dependent_composition(
                Dps=measured_Dp,
                measured_Ns=measured_N,
                N_modes=modes,
                splat_cutoff_nm=splat_cutoff_nm,
                splat_number_fraction=splat_number_fractions[model_species[spec]],
                mode_fractions=mode_fractions[line, spec*modes:spec*modes+modes],
                size_dist_params=params
            )
            total_Ns+=spec_Ns
            
            spec_cfg = {
                "type": "monodisperse",
                 "N": list(spec_Ns),
                 "D": list(measured_Dp),
                 "aero_spec_names": np.full((len(measured_Dp), 1), model_species[spec]).tolist(),
                 "aero_spec_fracs": np.ones((len(measured_Dp), 1)).tolist(),
            }
            species_pop = build_population(spec_cfg)
            spec_masses[model_species[spec]]=mass_thresholds[model_species[spec]][0][1]*np.sum(species_pop.spec_masses[:, species_pop.get_species_idx(model_species[spec])]*species_pop.num_concs) # kg/m^3
            total_mass += mass_thresholds[model_species[spec]][0][1]*np.sum(species_pop.spec_masses[:, species_pop.get_species_idx(model_species[spec])]*species_pop.num_concs) # kg/m^3

        # sample the BC and dust
        params=[splat_number_fractions['BC'], Dpg_BC, sigma_BC]
        spec_Ns=size_dependent_composition(measured_Dp, measured_N, 1,
                                                splat_cutoff_nm, 
                                                splat_number_fractions[model_species[spec]], 
                                                [1.0], params)
        total_Ns+=spec_Ns        
        
        params=[splat_number_fractions['OIN'], Dpg_dust, sigma_dust]
        spec_Ns=size_dependent_composition(measured_Dp, measured_N, 1,
                                                splat_cutoff_nm, 
                                                splat_number_fractions[model_species[spec]], 
                                                [1.0], params)
        total_Ns+=spec_Ns
        
        # match the measured number concentration
        mult=np.sum(measured_N)/np.sum(total_Ns)
        total_Ns*=mult
        
        # get the calculated mass fraction
        calculated_mass_fraction={}
        for ii in ams_mass_fractions.keys():
            try:
                calculated_mass_fraction[ii]=spec_masses[ii]/total_mass
            except:
                calculated_mass_fraction[ii]=0.0

        # get the calculated Ntot, Vtot, and SAtot
        calculated_mean_size = np.average(measured_Dp, weights=total_Ns)
        calculated_Vtot = np.sum(total_Ns*(4/3)*np.pi*(0.5*measured_Dp)**3)
        calculated_SA = np.sum(total_Ns*4*np.pi*(0.5*measured_Dp)**2)

        # calculate RSS
        RSS=((calculated_mean_size - measured_mean_size)/measured_mean_size)**2\
            + ((calculated_SA - measured_SA)/measured_SA)**2\
            + ((calculated_Vtot - measured_Vtot)/measured_Vtot)**2
        for ii in ams_mass_fractions.keys():
            RSS+=((calculated_mass_fraction[ii]-ams_mass_fractions[ii])/ams_mass_fractions[ii])**2

        if RSS<min_RSS:
            min_RSS=RSS
            line_to_save=line
            mult_to_save=mult

    output={}
    for ii in range(len(model_species)):
        output[model_species[ii]]=mode_fractions[line_to_save, ii*modes:ii*modes+modes]
    return output, mult_to_save

def sample_particle_masses(ptype, mass_thresholds, rng=None, max_tries=10_000):
    """
    Returns (temp_names, temp_fracs) where temp_fracs sum to 1 (approximately).
    """
    rng = rng or np.random.default_rng()
    included_species = []
    included_mass = []
    # --- Rejection sampling for included mass/species ---
    mean = mass_thresholds[ptype][0][1]
    std  = mass_thresholds[ptype][0][2]
    min_mass = mass_thresholds[ptype][0][0]
    included_species = list(mass_thresholds[ptype][1])
    total_incl_mass = None
    for _ in range(max_tries):
        total_incl_mass = rng.normal(loc=mean, scale=std)
        if not np.isfinite(total_incl_mass) or total_incl_mass <= 0:
            continue
        incl_species_mass = np.random.rand(len(included_species))
        incl_species_mass = total_incl_mass*(incl_species_mass/np.sum(incl_species_mass))
        included_mass = incl_species_mass.tolist()
        if total_incl_mass < 1.0 and total_incl_mass > min_mass:
            break
    
    # add NH4 to balance charges
    if "SO4" in included_species and "NO3" in included_species:
        SO4_frac=included_mass[included_species.index("SO4")]
        NO3_frac=included_mass[included_species.index("NO3")]
        SO4_data=retrieve_one_species("SO4")
        NO3_data=retrieve_one_species("NO3")
        NH4_data=retrieve_one_species("NH4")
        included_mass.append(SO4_frac*((2*NH4_data.molar_mass)/SO4_data.molar_mass)+NO3_frac*(NH4_data.molar_mass/NO3_data.molar_mass))
        included_species.append("NH4")
    elif "NO3" in included_species:
        NO3_frac=included_mass[included_species.index("NO3")]
        NO3_data=retrieve_one_species("NO3")
        NH4_data=retrieve_one_species("NH4")
        included_mass.append(NO3_frac*(NH4_data.molar_mass/NO3_data.molar_mass))
        included_species.append("NH4")
    if "SO4" in included_species and "NH4" not in included_species:
        SO4_frac=included_mass[included_species.index("SO4")]
        SO4_data=retrieve_one_species("SO4")
        NH4_data=retrieve_one_species("NH4")
        included_mass.append(SO4_frac*((2*NH4_data.molar_mass)/SO4_data.molar_mass))
        included_species.append("NH4")
    
    if np.sum(included_mass)>1.0:
        included_mass/=np.sum(included_mass)
        total_incl_mass=np.sum(included_mass)
    
    # sample the other species
    other_species=[]
    for t in mass_thresholds.keys():
        if t not in included_species and t not in ['IEPOX_SOA', 'NO3']:
            other_species.append(t)

    remaining_target = 1.0 - float(np.sum(included_mass))
    remaining_species = []
    remaining_mass = []
    ssum = 0.0
    n_other = len(other_species)
    while ssum < remaining_target:
        spec_name = other_species[rng.integers(0, n_other)]
        m = remaining_target * rng.random()
        if spec_name in remaining_species:
            remaining_mass[remaining_species.index(spec_name)] += m
            ssum += m
        else:
            remaining_species.append(spec_name)
            remaining_mass.append(m)
            ssum += m
    remaining_mass = np.asarray(remaining_mass, dtype=float)
    remaining_mass *= remaining_target / remaining_mass.sum()
    
    # # --- Combine ---
    temp_names = included_species + remaining_species
    temp_fracs = np.concatenate([included_mass, remaining_mass])

    return temp_names, temp_fracs

def sample_particle_Dp_N(
    particles_to_sample, particle_types, measured_number_fractions, Dp_mid_m, Dp_lo_m, Dp_hi_m, N_m3, 
    mode_fractions, size_distribution_parameters,
    splat_cutoff_nm=85, N_multiplier=1.0, size_dist_grid=3, fill_bins=True):
    """
    Returns (Dp, N) of each particle, where Dp is in m and N is in m^-3.
    """
    # get the size distribution for each type
    particle_diameters=np.zeros(particles_to_sample)
    particle_num_concs=np.zeros(particles_to_sample)
    
    # sample mass fraction in each particle
    for particle_type in list(np.unique(np.array(particle_types))):
        if particle_type=='BC':
            Dpg = 110e-9
            sigma = 1.6
            params=[measured_number_fractions['BC'], Dpg, sigma]
            original_SizeDist=N_multiplier*size_dependent_composition(Dp_mid_m, N_m3, 1,
                                                    splat_cutoff_nm, 
                                                    measured_number_fractions[particle_type], 
                                                    [1.0], params)
        elif particle_type=='OIN':
            Dpg = 110e-9 # taken from accumulation mode of MAM4
            sigma = 1.6
            params=[measured_number_fractions['OIN'], Dpg, sigma]
            original_SizeDist=N_multiplier*size_dependent_composition(Dp_mid_m, N_m3, 1,
                                                    splat_cutoff_nm, 
                                                    measured_number_fractions[particle_type], 
                                                    [1.0], params)
        else:
            params=[]
            for mode in range(len(size_distribution_parameters)):
                params.append(measured_number_fractions[particle_type]*mode_fractions[particle_type][mode])
                params.append(size_distribution_parameters[mode][1])
                params.append(size_distribution_parameters[mode][2])
            original_SizeDist = N_multiplier*size_dependent_composition(Dp_mid_m, N_m3, len(size_distribution_parameters),
                                                    splat_cutoff_nm, 
                                                    measured_number_fractions[particle_type], 
                                                    mode_fractions[particle_type], params) # 1/m^3

        # make sparser size distribution grid
        if size_dist_grid > 1:
            n_blocks = len(N_m3) // size_dist_grid
            m = n_blocks * size_dist_grid
            N_b = N_m3[:m].reshape(n_blocks, size_dist_grid)
            SD_b = original_SizeDist[:m].reshape(n_blocks, size_dist_grid)
            measured_N = N_b.sum(axis=1)#*100**3
            SizeDist = SD_b.sum(axis=1)
            Dp_lowers = Dp_lo_m[:m].reshape(n_blocks, size_dist_grid)[:, 0]
            Dp_uppers = Dp_hi_m[:m].reshape(n_blocks, size_dist_grid)[:, -1]
        else:
            measured_N = N_m3
            Dp_uppers = Dp_hi_m
            Dp_lowers = Dp_lo_m
            SizeDist = original_SizeDist
        Dp_mids = Dp_lowers + 0.5 * (Dp_uppers - Dp_lowers)

        # sample particle diameters 
        idx=np.where(np.array([particle_types])==particle_type)[1]
        if fill_bins: # forces one particle in each size distribution bin
            n_bins = len(Dp_lowers)
            if len(idx) < len(Dp_uppers):
                raise ValueError(f"Number of {particle_type} particles to sample is {len(idx)}, but there are {len(Dp_uppers)} size bins! Either increase particles_to_sample, decrease size_dist_grid, or set fill_bins to False.")
            counts = np.ones(n_bins, dtype=int)
            remaining = len(idx) - n_bins
            extra = np.random.default_rng().multinomial(remaining, np.ones(n_bins) / n_bins)
            counts += extra
            sampled_Dps = []
            for low, high, c in zip(Dp_lowers, Dp_uppers, counts):
                sampled_Dps.extend(10**(np.random.default_rng().uniform(np.log10(low), np.log10(high), c)))
            np.random.shuffle(sampled_Dps)
        else: # does not force one particle in each size distribution bin
            sampled_Dps=10**(np.log10(np.min(Dp_uppers))+(np.log10(np.max(Dp_uppers))-np.log10(np.min(Dp_uppers)))*np.random.rand(len(idx)))

        # get number concentrations according to each diameter that was sampled
        sampled_Ns=np.interp(sampled_Dps, xp=Dp_mids, fp=SizeDist)

        # change number concentrations based on histogram
        for jj in range(0,len(Dp_lowers)):
            idx2=np.where(np.logical_and(sampled_Dps >= Dp_lowers[jj], sampled_Dps < Dp_uppers[jj]))
            N_in_bin = len(idx2[0])
            sampled_Ns[idx2[0]]/=N_in_bin

        # save the sampled values
        particle_diameters[idx]=sampled_Dps # m
        particle_num_concs[idx]=sampled_Ns # m^-3

    return particle_diameters, particle_num_concs

def mass_fraction_comparison(
    particle_population, measured_mass_fractions, measured_mass_fraction_errors, mass_thresholds,
    override_matching=False):
    
    # get measured and sampled mass fractions
    sampled_mass_conc={}
    total_sampled_mass = 0
    for particle_type in measured_mass_fractions.keys():
        sampled_mass_conc[particle_type]=0
        try:
            for species in mass_thresholds[particle_type][1]:
                sampled_mass_conc[particle_type]+=np.sum(particle_population.spec_masses[:,particle_population.get_species_idx(species)]*particle_population.num_concs)
        except:
            species=particle_type
            sampled_mass_conc[particle_type]+=np.sum(particle_population.spec_masses[:,particle_population.get_species_idx(species)]*particle_population.num_concs)
    
    for species in ['IEPOX_OS', 'tetrol', 'tetrol_olig', 'IEPOX_OH_SOA']:
        try:
            sampled_mass_conc['OC']+=np.sum(particle_population.spec_masses[:,particle_population.get_species_idx(species)]*particle_population.num_concs)
        except:
            pass

    # normalize
    total_sampled_mass = sum(sampled_mass_conc.values()) # np.sum(np.sum(particle_population.spec_masses, axis=1)*particle_population.num_concs) 
    sampled_mass_fractions = {}
    for kk in sampled_mass_conc.keys():
        sampled_mass_fractions[kk] = sampled_mass_conc[kk]/total_sampled_mass
    
    # check that the mass fractions match the AMS measurement
    checks=[]
    for group in measured_mass_fractions.keys():
        sampled=sampled_mass_fractions[group]
        measured=measured_mass_fractions[group]
        measured_error=measured_mass_fraction_errors[group]
        if sampled >= measured-measured_error and sampled <= measured+measured_error:
            checks.append(True)
        else:
            checks.append(False)

    if override_matching==True:
        for ii in range(len(checks)):
            checks[ii]=True

    return sampled_mass_fractions, checks

def classify_particles(particle_population, mass_thresholds):
    """ Classifies particles according to input mass thresholds."""
    particle_classes = np.zeros(len(particle_population.num_concs), dtype='U20')
    for group in mass_thresholds.keys():
        spec_fracs = np.zeros(len(particle_population.num_concs))
        for species in mass_thresholds[group][1]:
            spec_fracs+=particle_population.spec_masses[:,particle_population.get_species_idx(species)]/np.sum(particle_population.spec_masses, axis=1)
        spec_idx = np.where(spec_fracs>=mass_thresholds[group][0][0])[0]
        for ii in spec_idx: # doesn't overwrite particles that have already had a class assigned
            if particle_classes[ii]=='':
                particle_classes[ii]=group
    return particle_classes

def number_fraction_comparison(
    particle_population, mass_thresholds, measured_number_fractions, 
    measured_number_fraction_errors, splat_cutoff_nm=85):
    """ Compares sampled population with the miniSPLAT measurements. """
    particle_classes = classify_particles(particle_population, mass_thresholds)
    particle_diameters = particle_population.get_particle_var('Ddry')
    particle_num_concs = particle_population.num_concs
    all_idx=np.where(particle_diameters>=splat_cutoff_nm*1e-9)
    sampled_number_fractions = {}
    checks = []
    for species in mass_thresholds.keys():
        spec_idx=np.where((particle_classes==species) & (particle_diameters>=splat_cutoff_nm*1e-9))
        sampled=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
        measured=measured_number_fractions[species]
        measured_error=measured_number_fraction_errors[species]
        sampled_number_fractions[species]=sampled
        if sampled >= measured-measured_error and sampled <= measured+measured_error:
            checks.append(True)
        else:
            checks.append(False)
    
    return sampled_number_fractions, checks

def _plot_size_distribution(
    particle_population, mass_thresholds, Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3,
    outpath, size_dist_grid=3):
    """ plot the measured and sampled size distribution. """
    
    # make sparser size distribution grid
    if size_dist_grid > 1:
        n_blocks = len(N_cm3) // size_dist_grid
        m = n_blocks * size_dist_grid
        N_b = N_cm3[:m].reshape(n_blocks, size_dist_grid)
        N_b_error = N_std_cm3[:m].reshape(n_blocks, size_dist_grid)
        measured_N = N_b.sum(axis=1)
        measured_N_error = N_b_error.sum(axis=1)
        Dp_lowers = Dp_lo_nm[:m].reshape(n_blocks, size_dist_grid)[:, 0]
        Dp_uppers = Dp_hi_nm[:m].reshape(n_blocks, size_dist_grid)[:, -1]
    else:
        measured_N = N_cm3
        measured_N_error = N_std_cm3
        Dp_uppers = Dp_hi_nm
        Dp_lowers = Dp_lo_nm
    Dp_mids = Dp_lowers + 0.5 * (Dp_uppers - Dp_lowers)

    # get particle classes
    particle_classes = classify_particles(particle_population, mass_thresholds)
    bottom=np.zeros(len(Dp_uppers)-1)
    plt.errorbar(Dp_mids, measured_N/np.max(measured_N), fmt='o', yerr=measured_N_error/np.max(measured_N), mfc='w', mec='k', ecolor='k')
    
    particle_diameters_nm = 1e9*particle_population.get_particle_var('Ddry')
    particle_num_concs_cm3 = 1e-6*particle_population.num_concs
    hist=np.histogram(particle_diameters_nm, bins=Dp_uppers, weights=particle_num_concs_cm3)
    hist_max=np.max(hist[0])
    for t, c in zip(['BC','OIN','SO4','NO3','OC','IEPOX_SOA'], ['grey','gold','r','b','g','C6']):
        idx=np.where(particle_classes==t)[0]
        hist=np.histogram(particle_diameters_nm[idx], bins=Dp_uppers, weights=particle_num_concs_cm3[idx])
        widths=hist[1][1:]-hist[1][:-1]
        plt.bar(Dp_uppers[:-1], hist[0]/hist_max, width=widths, align='edge', bottom=bottom, facecolor=c, edgecolor='k', label=t)
        bottom+=hist[0]/hist_max
    plt.xscale('log')
    plt.ylabel(r'Normalized Number Concentration (cm$^{-3}$)', labelpad=10)
    plt.xlabel('Dry Diameter (nm)', labelpad=10)
    plt.legend()
    plt.ylim(0,)
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

    return

def _plot_bar_compare(d_true: dict, d_pop: dict, title: str, outpath: str):

    keys = sorted(set(d_true.keys()) | set(d_pop.keys()))
    true = np.array([d_true.get(k, 0.0) for k in keys], dtype="float64")
    pop = np.array([d_pop.get(k, 0.0) for k in keys], dtype="float64")

    x = np.arange(len(keys))
    w = 0.40

    plt.figure()
    plt.bar(x - w / 2, true, width=w, label="Observed")
    plt.bar(x + w / 2, pop, width=w, label="Reconstructed")
    plt.xticks(x, keys, rotation=45, ha="right")
    plt.ylabel("Fraction")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return







# -----------------------------
# Builder entrypoint
# -----------------------------

@register("hiscale_observations")
def build(config: Dict[str, Any]) -> ParticlePopulation:
    config = normalize_population_config(config)
    required = ["aimms_file", "splat_file", "ams_file", "z", "dz", "splat_species", "mass_thresholds"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"hiscale_observations missing required config keys: {missing}")
    if "fims_file" not in config and "beasd_file" not in config:
        raise KeyError("hiscale_observations requires either 'fims_file' or 'beasd_file' in config.")
    if "fims_file" in config and "fims_bins_file" not in config:
        raise KeyError("hiscale_observations with 'fims_file' requires 'fims_bins_file' in config.")

    if "fims_file" in config:
        fims_file = str(config["fims_file"])
        fims_bins_file = str(config["fims_bins_file"])
        fims_time_col = config.get("fims_time_col", None)      # usually None; auto-detect
        fims_cloud_col = str(config.get("fims_cloud_col", "Cloud_flag"))
        fims_density_measure = str(config.get("fims_density_measure", "ln"))
    elif "beasd_file" in config:
        beasd_file = str(config["beasd_file"])
        beasd_time_col = config.get("beasd_time_col", None)      # usually None; auto-detect
        beasd_cloud_col = str(config.get("beasd_cloud_col", "Cloud_flag"))
        beasd_density_measure = str(config.get("beasd_density_measure", "per_bin"))

    aimms_file = str(config["aimms_file"])
    splat_file = str(config["splat_file"])
    ams_file = str(config["ams_file"])
    N_particles = int(config.get("N_particles", 1000))
    z = float(config["z"])
    dz = float(config["dz"])
    splat_species = dict(config["splat_species"])
    mass_thresholds = dict(config["mass_thresholds"])
    size_dist_grid = int(config.get("size_dist_grid", 3))
    preferred_matching = str(config.get("preferred_matching", "mass")) # can be either mass or number, determines how the number concentration gets scaled
    cloud_flag_value = config.get("cloud_flag_value", 0)
    max_dp_nm = float(config.get("max_dp_nm", 1000.0))
    splat_cutoff_nm = float(config.get("splat_cutoff_nm", 0.0))
    region_filter = config.get("region_filter", None)
    fill_bins = config.get("fill_bins", True)

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

    # --- read average size distribution ---
    if "fims_file" in config:
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
        size_dist_type="FIMS"
        size_dist_file=fims_file
        size_dist_time_col=fims_time_col
        size_dist_cloud_col=fims_cloud_col
        size_dist_density_measure=fims_density_measure
    
    elif "beasd_file" in config:
        Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3 = _read_beasd_avg_size_dist(
            beasd_file=beasd_file,
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
            beasd_time_col=beasd_time_col,
            beasd_cloud_col=beasd_cloud_col,
            beasd_density_measure=beasd_density_measure,
        )
        size_dist_type="BEASD"
        size_dist_file=beasd_file
        size_dist_time_col=beasd_time_col
        size_dist_cloud_col=beasd_cloud_col
        size_dist_density_measure=beasd_density_measure
    
    Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)

    # --- miniSPLAT reduced-class number fractions ---
    type_fracs, type_fracs_err = _read_minisplat_number_fractions(
        splat_file=splat_file,
        aimms_file=aimms_file,
        size_dist_type=size_dist_type,
        size_dist_file=size_dist_file,
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
        size_dist_time_col=size_dist_time_col,
        size_dist_cloud_col=size_dist_cloud_col,
    )
    tf = _normalize_fracs(type_fracs)
    if not tf:
        raise RuntimeError("miniSPLAT type fractions sum to 0; cannot build population.")

    # --- AMS mass fractions + measured total mass (for optional scaling) ---
    ams_mass_frac, ams_mass_frac_err, measured_mass, measured_mass_err = _read_ams_mass_fractions(
        ams_file=ams_file,
        aimms_file=aimms_file,
        size_dist_type=size_dist_type,
        size_dist_file=size_dist_file,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=size_dist_time_col,
        size_dist_cloud_col=size_dist_cloud_col,
        ams_time_col=config.get("ams_time_col", None),
        ams_flag_col=config.get("ams_flag_col", None),
    )  

    # --- units ---
    # N_cm3 is per-bin number concentration (cm^-3) returned by _read_fims_avg_size_dist.
    # Convert to m^-3 for particle allocation and derive dN/dlnD for diagnostics/metadata.
    Dp_mid_m = Dp_mid_nm * 1e-9
    Dp_lo_m = Dp_lo_nm * 1e-9
    Dp_hi_m = Dp_hi_nm * 1e-9
    N_m3 = N_cm3 * 1e6  # cm^-3 -> m^-3
    dln = np.log(Dp_hi_nm / Dp_lo_nm)
    if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
        raise ValueError("Invalid FIMS bin edges; cannot compute dln widths.")
    dNdln_m3 = N_m3 / dln

    # break the size distribution into N modes and provide fitting parameters
    size_distribution_pars = fit_Nmodal_distibution(Dp_mid_m, N_m3)   
    
    # move the splat species into the different modes to optimize matching with the size distribution and measured mass fractions
    mode_fractions, N_multiplier = optimize_splat_species_distributions(
        splat_species=splat_species,
        size_distribution_pars=size_distribution_pars,
        measured_Dp=Dp_mid_m,
        measured_N=N_m3,
        splat_number_fractions=tf,
        ams_mass_fractions=ams_mass_frac,
        splat_cutoff_nm=splat_cutoff_nm,
        mass_thresholds=mass_thresholds,
        datapoints=1000
    )

    # set up checks
    checks=[False]
    counter = 0
    maxcounter=100
    particle_species=list(tf.keys())
    while (sum(checks)!=len(checks) and counter<maxcounter):
        
        for _ in range(100):
            try:
                # sample which particles are which 
                ptypes = []
                for species in particle_species:
                    ptypes.extend([species] * 1)
                remaining = N_particles - len(ptypes)
                if remaining > 0:
                    ptypes.extend(np.random.choice(particle_species, size=remaining).tolist())
                np.random.shuffle(ptypes)

                # sample the type-dependent size and concentration of each particle
                particle_diameters, particle_num_concs = sample_particle_Dp_N(
                    N_particles, ptypes, tf, Dp_mid_m, Dp_lo_m, Dp_hi_m, N_m3, 
                    mode_fractions, size_distribution_pars, splat_cutoff_nm=splat_cutoff_nm,
                    N_multiplier=N_multiplier, fill_bins=fill_bins, size_dist_grid=size_dist_grid)
                break   
            
            except:
                pass
        else:
            raise ValueError(f"Sampling unsuccessful when fill_bins = {fill_bins}.")

        # change number concentrations to match measurements
        mult={}
        for ptype in tf.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==ptype, particle_diameters>=splat_cutoff_nm*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff_nm*1e-9)
            modeled_Nfraction=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
            mult[ptype]=tf[ptype]/modeled_Nfraction
        for ptype in tf.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==ptype, particle_diameters>=splat_cutoff_nm*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff_nm*1e-9)
            particle_num_concs[spec_idx[0]]*=mult[ptype]
            modeled_Nfraction=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])

        # build a map of mass fractions
        species_map = []
        for ptype in mass_thresholds.keys():
            for species in mass_thresholds[ptype][1]:
                species_map.append(species)
        if "SO4" in species_map or "NO3" in species_map:
            if "NH4" not in species_map:
                species_map.append("NH4")
        species_map = np.asarray(species_map)

        # sample the mass fraction of species in each particle
        aero_spec_fracs=np.zeros((len(ptypes), len(species_map)))
        for ii in range(len(ptypes)):
            temp_names, temp_fracs = sample_particle_masses(ptypes[ii], mass_thresholds, rng=None, max_tries=10_000)
            for name, frac in zip(temp_names, temp_fracs):            
                try:
                    jj = np.where(species_map==name)[0][0]
                except:
                    raise ValueError(f"No place in species map for {name}.")
                aero_spec_fracs[ii,jj]=frac
        aero_spec_names = np.tile(species_map, (N_particles, 1))

        # make the particle population from the list of species and mass fractions
        population_cfg={
            "type": "monodisperse",
            "N": particle_num_concs,
            "D": particle_diameters,
            "aero_spec_names": aero_spec_names,
            "aero_spec_fracs": aero_spec_fracs,
            "species_modifications": species_modifications}
        particle_population = build_population(population_cfg)

        # check if the bulk mass fraction matches measurements
        sampled_mass_fractions, mass_fraction_checks = mass_fraction_comparison(
            particle_population, ams_mass_frac, ams_mass_frac_err, mass_thresholds)
        
        # check that particles match miniSPLAT measurements
        sampled_number_fractions, number_fraction_checks = number_fraction_comparison(
            particle_population, mass_thresholds, tf, type_fracs_err, splat_cutoff_nm=splat_cutoff_nm)

        # combine the checks
        checks = mass_fraction_checks+number_fraction_checks
        counter += 1
    
    # change the number concentrations so that the total
    # mass concentration matches the AMS measurements or the 
    # total number concentration matches the FIMS/BEASD
    if preferred_matching=="mass":
        total_mass = np.sum(np.sum(particle_population.spec_masses, axis=1)*particle_population.num_concs)
        particle_population.num_concs*=measured_mass/(1e9*total_mass)
    elif preferred_matching=="number":
        particle_population.num_concs*=np.nansum(N_m3)/np.sum(particle_population.num_concs)
    else:
        raise ValueError(f"preferred_matching must be 'number' or 'mass', got {preferred_matching}.")

    # output diagnostics
    outdir=str(config.get("outdir", "."))
    prefix=str(config.get("prefix", ""))
    summary = Path(outdir) / f"{prefix}diagnostics_summary.txt"
    with summary.open("w") as f:
        f.write("Diagnostics summary\n")
        f.write("===================\n\n")

        f.write("Fitted size distribution (Dpg_nm, sigma):\n")
        for ii, pars in enumerate(size_distribution_pars):
            f.write(f"  Mode {ii}: {float(pars[1]*1e9):.6g}, {float(pars[2]):.4g}\n")
        f.write("\n")
        
        f.write("Optimized fraction of particles in each mode:\n")
        for spec in mode_fractions:
            f.write(f"  {spec}: ")
            for ii in range(len(size_distribution_pars)):
                f.write(f"{float(mode_fractions[spec][ii]):.3g} ")
            f.write(f"\n")
        f.write("\n")

        f.write(f"Total N (FIMS, cm^-3): {float(np.nansum(N_cm3)):.6g}\n")
        f.write(f"Total N (pop,  cm^-3): {float(np.nansum(1e-6*np.sum(particle_population.num_concs))):.6g}\n")
        if float(np.nansum(N_cm3)) > 0:
            f.write(f"Ratio (pop/FIMS):      {float(np.nansum(1e-6*np.sum(particle_population.num_concs)))/float(np.nansum(N_cm3)):.6g}\n")
        f.write("\n")

        f.write("AMS mass fractions (observed, normalized over available keys):\n")
        for kk in ams_mass_frac.keys():
            v = ams_mass_frac.get(kk, 0.0)
            f.write(f"  {kk}: {v:.4f}\n")

        f.write("\nAMS mass fractions (reconstructed, normalized over available keys):\n")
        for kk in ams_mass_frac:
            v = sampled_mass_fractions.get(kk, 0.0)
            f.write(f"  {kk}: {v:.4f}\n")

        f.write("\nminiSPLAT number fractions (observed, normalized):\n")
        for kk in tf.keys():
            v = tf.get(kk, 0.0)
            f.write(f"  {kk}: {v:.4f}\n")

        f.write("\nminiSPLAT number fractions (reconstructed, normalized):\n")
        for kk in tf.keys():
            v = sampled_number_fractions.get(kk, 0.0)
            f.write(f"  {kk}: {v:.4f}\n")

        
    # plot the size distribution
    _plot_size_distribution(
        particle_population, mass_thresholds, Dp_lo_nm, Dp_hi_nm, 
        N_cm3, N_std_cm3, Path(outdir) / f"{prefix}diag_size_dist.png", 
        size_dist_grid=size_dist_grid)

    # plot the number fractions
    _plot_bar_compare(
        tf, sampled_number_fractions, 
        "miniSPLAT number fractions (observed vs reconstructed)", 
        Path(outdir) / f"{prefix}diag_minisplat_type_fracs.png")
    
    # plot mass fractions
    _plot_bar_compare(
        ams_mass_frac,
        sampled_mass_fractions,
        "AMS bulk mass fractions (observed vs reconstructed)",
        Path(outdir) / f"{prefix}diag_ams_mass_fracs.png",
    )

    # Save particle population metadata
    particle_population.metadata = {
        "source": "hiscale_observations",
        "z": z,
        "dz": dz,
        "cloud_flag_value": cloud_flag_value,
        "max_dp_nm": max_dp_nm,
        "splat_cutoff_nm": splat_cutoff_nm,
        "size_distribution_density_measure": size_dist_density_measure,
        "type_fracs": tf,
        "type_fracs_err": type_fracs_err,
        "ams_mass_frac": ams_mass_frac,
        "ams_mass_frac_err": ams_mass_frac_err,
        "measured_ams_total_mass": measured_mass,
        "measured_ams_total_mass_err": measured_mass_err,
        "size_distribution": {
            "Dp_lo_nm": Dp_lo_nm.copy(),
            "Dp_hi_nm": Dp_hi_nm.copy(),
            "dln": dln.copy(),
            "N_bin_m3": N_m3.copy(),
            "dNdlnD_m3": dNdln_m3.copy(),
        },
    }

    return particle_population
