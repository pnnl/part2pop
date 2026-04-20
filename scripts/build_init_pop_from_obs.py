#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an initial population from HI-SCALE observations using part2pop's
hiscale_observations builder, with robust date-driven file selection.

Usage:
  python scripts/build_init_pop_from_obs.py \
    --root /path/to/multipart_archived/separate_tools/datasets \
    --date 20160425 --z 100 --dz 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from datetime import datetime
from typing import Iterable, List, Dict

import numpy as np

from part2pop.population import build_population
# note: these are "private" convenience readers provided by the hiscale builder module.
from part2pop.population.factory.hiscale_observations import (
    _read_fims_avg_size_dist,
    _read_minisplat_number_fractions,
    _read_ams_mass_fractions,
)

from part2pop.analysis.builder import build_variable


# -----------------------------
# Small utilities
# -----------------------------
def _parse_date(date_str: str) -> datetime:
    """Accepts 'YYYYMMDD' or 'YYYY-MM-DD'."""
    date_str = date_str.strip()
    if re.fullmatch(r"\d{8}", date_str):
        return datetime.strptime(date_str, "%Y%m%d")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return datetime.strptime(date_str, "%Y-%m-%d")
    raise ValueError(f"Unrecognized date format: {date_str}. Use YYYYMMDD or YYYY-MM-DD.")


def _pick_one(paths: Iterable[Path], label: str) -> Path:
    paths = sorted([Path(p) for p in paths])
    if len(paths) == 0:
        raise FileNotFoundError(f"Could not find any file for {label}.")
    if len(paths) == 1:
        return paths[0]
    print(f"WARNING: multiple candidates for {label}; using {paths[-1].name}")
    return paths[-1]


def _extract_dates_from_names(paths: Iterable[Path]) -> List[str]:
    """
    Extract YYYYMMDD occurrences from filenames to help with diagnostics.
    """
    dates = set()
    pat = re.compile(r"(20\d{6})")  # catches 20160425 etc.
    for p in paths:
        m = pat.search(p.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates)


# -----------------------------
# Path resolution logic
# -----------------------------
def resolve_hiscale_paths(datasets_root: Path, date: datetime) -> Dict[str, Path]:
    """
    Resolve files under datasets_root for a given date.

    We prefer the directory:
      datasets_root/HISCALE_data_MMDD

    If that directory does not exist, we search all HISCALE_data_* directories
    for ANY file containing the YYYYMMDD string for the requested date.
    """
    datasets_root = Path(datasets_root).expanduser().resolve()
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets_root not found: {datasets_root}")

    mmdd = date.strftime("%m%d")
    yyyymmdd = date.strftime("%Y%m%d")
    yyyy = date.strftime("%Y")
    day = date.strftime("%d").lstrip("0")
    mon_abbrev = date.strftime("%b")  # Apr, May, ...

    preferred = datasets_root / f"HISCALE_data_{mmdd}"

    # 1) pick dataset_dir
    if preferred.exists() and preferred.is_dir():
        dataset_dir = preferred
    else:
        # Search all HISCALE_data_* directories for any file containing the date string
        candidates = []
        for d in sorted(datasets_root.glob("HISCALE_data_*")):
            if not d.is_dir():
                continue
            if any(d.glob(f"*{yyyymmdd}*")):
                candidates.append(d)

        if len(candidates) == 0:
            # Provide diagnostic: what dates are present?
            all_files = list(datasets_root.glob("HISCALE_data_*/*"))
            dates_present = _extract_dates_from_names(all_files)
            msg = (
                f"No HISCALE_data_* directory under {datasets_root} contains any file with date '{yyyymmdd}'.\n"
                f"Dates present (from filenames, sample): {dates_present[:20]}"
                + (" ..." if len(dates_present) > 20 else "")
            )
            raise FileNotFoundError(msg)

        if len(candidates) > 1:
            print("WARNING: multiple dataset dirs contain the requested date; using:", candidates[-1].name)
        dataset_dir = candidates[-1]

    # 2) resolve instrument files within dataset_dir using expected patterns
    fims_candidates = list(dataset_dir.glob(f"FIMS_G1_{yyyymmdd}_R*_HISCALE_001s.txt"))
    aimms_candidates = list(dataset_dir.glob(f"AIMMS20_G1_{yyyymmdd}*_HISCALE020h.txt"))
    ams_candidates = list(dataset_dir.glob(f"HiScaleAMS_G1_{yyyymmdd}_R*.txt"))
    splat_candidates = list(dataset_dir.glob(f"Splat_Composition_{day}-{mon_abbrev}-{yyyy}.txt"))

    # broader fallbacks
    if len(fims_candidates) == 0:
        fims_candidates = list(dataset_dir.glob(f"FIMS*_G1*{yyyymmdd}*HISCALE*001s*.txt"))
    if len(aimms_candidates) == 0:
        aimms_candidates = list(dataset_dir.glob(f"AIMMS*G1*{yyyymmdd}*HISCALE020h*.txt"))
    if len(ams_candidates) == 0:
        ams_candidates = list(dataset_dir.glob(f"HiScaleAMS*G1*{yyyymmdd}*R*.txt"))
    if len(splat_candidates) == 0:
        splat_candidates = list(dataset_dir.glob("Splat_Composition_*.txt"))

    missing = []
    if len(fims_candidates) == 0:
        missing.append("FIMS")
    if len(aimms_candidates) == 0:
        missing.append("AIMMS")
    if len(ams_candidates) == 0:
        missing.append("AMS")
    if len(splat_candidates) == 0:
        missing.append("SPLAT")

    if missing:
        listing = sorted([p.name for p in dataset_dir.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f"Missing required files in {dataset_dir} for date {yyyymmdd}: {missing}\n"
            f"Directory listing:\n  " + "\n  ".join(listing[:80]) + ("\n  ..." if len(listing) > 80 else "")
        )

    fims_file = _pick_one(fims_candidates, "FIMS")
    aimms_file = _pick_one(aimms_candidates, "AIMMS")
    ams_file = _pick_one(ams_candidates, "AMS")
    splat_file = _pick_one(splat_candidates, "miniSPLAT composition")

    bins_candidates = list(dataset_dir.glob("HISCALE_FIMS_bins_R*.txt"))
    if len(bins_candidates) == 0:
        # also allow any file that looks like bins if the strict name isn't present
        bins_candidates = list(dataset_dir.glob("*bins*.txt")) + list(dataset_dir.glob("*FIMS*bin*.txt"))

    bins_file = _pick_one(bins_candidates, "FIMS bins")

    return {
        "dataset_dir": dataset_dir,
        "fims_file": fims_file,
        "aimms_file": aimms_file,
        "ams_file": ams_file,
        "splat_file": splat_file,
        "fims_bins_file": bins_file,
    }


# -----------------------------
# Diagnostics
# -----------------------------
def _plot_size_dist_compare(Dp_lo_nm, Dp_hi_nm, dNdln_fims_m3, dNdln_pop_m3, outpath):
    import matplotlib.pyplot as plt

    Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)
    
    plt.figure()
    plt.plot(Dp_mid_nm, dNdln_fims_m3, marker="o", linestyle="-", label="FIMS (avg, filtered)")
    plt.plot(Dp_mid_nm, dNdln_pop_m3, marker="s", linestyle="--", label="Reconstructed pop")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("Dp (nm)")
    plt.ylabel(r"$dN/d\ln D$ (m$^{-3}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_bar_compare(d_true: dict, d_pop: dict, title: str, outpath: str):
    import matplotlib.pyplot as plt

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

def make_diagnostics(
    *,
    pop,
    Dp_lo_nm,
    Dp_hi_nm,
    N_cm3,
    type_fracs_obs,
    ams_mass_frac_obs,
    outdir=".",
    prefix="",
):
    """
    Diagnostics:
      1) FIMS measured size distribution vs reconstructed from pop (dN/dlnD on FIMS bins)
      2) AMS bulk mass fractions: observed vs reconstructed from population (N-weighted species mass)
      3) miniSPLAT number fractions: observed vs reconstructed (from pop.metadata["type_fracs"] if available)

    Assumptions:
      - pop.num_concs are in m^-3
      - pop species masses are available as pop.spec_masses (per-particle, kg) aligned with pop.species
      - particle diameter is accessible via get_Ddry()/get_Dcore() or a common Dp-like attribute
    """
    from pathlib import Path
    import numpy as np

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ids = list(pop.ids)
    num_concs = np.asarray(pop.num_concs, dtype=float)
    N_fims_total_m3 = float(np.nansum(N_cm3) * 1e6)
    N_pop_total_m3 = float(pop.get_Ntot())

    # -----------------------
    # Helpers
    # -----------------------
    def _norm(d):
        d = {k: float(v) for k, v in (d or {}).items() if np.isfinite(v) and float(v) > 0}
        s = sum(d.values())
        return {k: v / s for k, v in d.items()} if s > 0 else {}

    # -----------------------
    # 1) Size distribution: use analysis variable builder for dN/dlnD on FIMS bins
    # -----------------------
    Dp_lo_nm = np.asarray(Dp_lo_nm, dtype=float)
    Dp_hi_nm = np.asarray(Dp_hi_nm, dtype=float)
    N_cm3 = np.asarray(N_cm3, dtype=float)
    if not (Dp_lo_nm.size and Dp_lo_nm.size == Dp_hi_nm.size == N_cm3.size):
        raise ValueError("FIMS inputs must be equal-length arrays.")
    Dp_lo_m = Dp_lo_nm * 1e-9
    Dp_hi_m = Dp_hi_nm * 1e-9

    edges_m = np.concatenate([Dp_lo_m[:1], Dp_hi_m])  # len nbins+1
    dln = np.log(Dp_hi_m / Dp_lo_m)
    if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
        raise ValueError("Invalid log bin widths derived from FIMS bins.")

    # Convert FIMS per-bin counts (cm^-3) → dN/dlnD in m^-3
    dNdln_fims_m3 = (N_cm3 * 1e6) / dln
    fims_from_hist = float(np.nansum(dNdln_fims_m3 * dln))
    if fims_from_hist > 0 and np.isfinite(fims_from_hist) and np.isfinite(N_fims_total_m3):
        fims_scale = N_fims_total_m3 / fims_from_hist
        if not np.isclose(fims_scale, 1.0, rtol=1e-4, atol=0):
            dNdln_fims_m3 *= fims_scale

    # Population dN/dlnD via analysis variable; use same edges for one-to-one comparison
    dNdln_var = build_variable(
        "dNdlnD",
        scope="population",
        var_cfg={
            "edges": edges_m,
            "method": "hist",
            "wetsize": False,  # match previous dry-diameter binning
        },
    )
    pop_dist = dNdln_var.compute(population=pop, as_dict=True)
    dNdln_pop_m3 = np.asarray(pop_dist["dNdlnD"], dtype=float)
    if dNdln_pop_m3.shape[0] != dNdln_fims_m3.shape[0]:
        raise ValueError("Computed dN/dlnD does not align with FIMS bins (check edges).")
    pop_from_hist = float(np.nansum(dNdln_pop_m3 * dln))
    if pop_from_hist > 0 and np.isfinite(pop_from_hist) and np.isfinite(N_pop_total_m3):
        pop_scale = N_pop_total_m3 / pop_from_hist
        if not np.isclose(pop_scale, 1.0, rtol=1e-4, atol=0):
            dNdln_pop_m3 *= pop_scale

    # For summaries we still want per-bin counts [cm^-3]
    N_pop_cm3 = (dNdln_pop_m3 * dln) / 1e6

    _plot_size_dist_compare(
        Dp_lo_nm,
        Dp_hi_nm,
        dNdln_fims_m3,
        dNdln_pop_m3,
        outdir / f"{prefix}diag_size_dist.png",
    )

    # -----------------------
    # 2) AMS mass fractions: observed vs reconstructed from population
    # -----------------------
    # Define the AMS species we care about for plotting/summary
    ams_keys = ["SO4", "NO3", "OC", "NH4"]

    # Observed (normalize defensively)
    obs_ams = {k: float(ams_mass_frac_obs.get(k, np.nan)) for k in ams_keys}
    # If user passed unnormalized fractions, normalize over finite positives
    obs_ams_norm = _norm({k: v for k, v in obs_ams.items() if np.isfinite(v)})

    # Reconstructed from pop.spec_masses if available
    pop_ams = {k: np.nan for k in ams_keys}
    try:
        species_names = [s.name for s in pop.species]
        spec_masses = np.asarray(pop.spec_masses, dtype=float)  # (n_particles, n_species)
        if spec_masses.ndim != 2 or spec_masses.shape[0] != len(ids):
            raise ValueError(
                f"Unexpected pop.spec_masses shape {spec_masses.shape}; expected ({len(ids)}, n_species)"
            )

        # total mass per species [kg m^-3]
        mass_kg_m3 = (num_concs[:, None] * spec_masses).sum(axis=0)
        tot = float(np.nansum(mass_kg_m3))
        if tot > 0:
            mass_frac = mass_kg_m3 / tot
            for k in ams_keys:
                if k in species_names:
                    pop_ams[k] = float(mass_frac[species_names.index(k)])

        pop_ams_norm = _norm({k: v for k, v in pop_ams.items() if np.isfinite(v)})
    except Exception:
        # Keep NaNs; still write size/type plots and summary
        pop_ams_norm = {}

    # Plot AMS comparison (use normalized dicts; missing keys become 0)
    _plot_bar_compare(
        obs_ams_norm,
        pop_ams_norm,
        "AMS bulk mass fractions (observed vs reconstructed)",
        outdir / f"{prefix}diag_ams_mass_fracs.png",
    )

    # -----------------------
    # 3) miniSPLAT number fractions: observed vs builder metadata
    # -----------------------
    md = getattr(pop, "metadata", None)
    pop_tf_raw = {}
    if isinstance(md, dict) and "type_fracs" in md and isinstance(md["type_fracs"], dict):
        pop_tf_raw = md["type_fracs"]

    obs_tf = _norm(type_fracs_obs)
    pop_tf = _norm(pop_tf_raw)

    _plot_bar_compare(
        obs_tf,
        pop_tf,
        "miniSPLAT number fractions (observed vs reconstructed)",
        outdir / f"{prefix}diag_minisplat_type_fracs.png",
    )

    # -----------------------
    # Summary text
    # -----------------------
    summary = outdir / f"{prefix}diagnostics_summary.txt"
    with summary.open("w") as f:
        f.write("Diagnostics summary\n")
        f.write("===================\n\n")

        f.write(f"Total N (FIMS, cm^-3): {float(np.nansum(N_cm3)):.6g}\n")
        f.write(f"Total N (pop,  cm^-3): {float(np.nansum(N_pop_cm3)):.6g}\n")
        if float(np.nansum(N_cm3)) > 0:
            f.write(f"Ratio (pop/FIMS):      {float(np.nansum(N_pop_cm3))/float(np.nansum(N_cm3)):.6g}\n")
        f.write("\n")

        f.write("AMS mass fractions (observed, normalized over available keys):\n")
        for k in ams_keys:
            v = obs_ams_norm.get(k, 0.0)
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\nAMS mass fractions (reconstructed, normalized over available keys):\n")
        for k in ams_keys:
            v = pop_ams_norm.get(k, 0.0)
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\nminiSPLAT number fractions (observed, normalized):\n")
        for k, v in sorted(obs_tf.items()):
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\nminiSPLAT number fractions (reconstructed / builder metadata, normalized):\n")
        if pop_tf:
            for k, v in sorted(pop_tf.items()):
                f.write(f"  {k}: {v:.4f}\n")
        else:
            f.write("  (no pop.metadata['type_fracs'] available)\n")

# def make_diagnostics(
#     *,
#     pop,
#     Dp_lo_nm,
#     Dp_hi_nm,
#     N_cm3,
#     type_fracs_obs,
#     ams_mass_frac_obs,
#     outdir=".",
#     prefix="",
# ):
#     """
#     Diagnostics:
#       1) FIMS measured size distribution vs reconstructed from pop
#       2) AMS mass fractions: observed vs reconstructed
#       3) miniSPLAT number fractions: observed vs reconstructed

#     Assumptions:
#       - pop.num_concs are in m^-3
#       - pop particles have diameter accessible as particle.get_Ddry()/get_Dp()/Dp attribute, etc.
#       - pop.spec_masses are per-particle species masses in kg aligned to pop.species ordering.
#     """
#     from pathlib import Path
#     import matplotlib.pyplot as plt

#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     # --- Size distribution: FIMS bins vs reconstructed from population (direct binning) ---
#     Dp_lo_nm = np.asarray(Dp_lo_nm, dtype=float)
#     Dp_hi_nm = np.asarray(Dp_hi_nm, dtype=float)
#     N_cm3    = np.asarray(N_cm3, dtype=float)

#     Dp_lo_m = Dp_lo_nm * 1e-9
#     Dp_hi_m = Dp_hi_nm * 1e-9

#     # bin edges in meters (len = nbins+1)
#     edges_m = np.concatenate([Dp_lo_m[:1], Dp_hi_m])
#     nb = len(Dp_lo_m)

#     # reconstructed per-bin number concentration (m^-3 per bin)
#     N_pop_m3 = np.zeros(nb, dtype=float)

#     # iterate particles and bin by dry diameter
#     ids = list(pop.ids)
#     num_concs = np.asarray(pop.num_concs, dtype=float)

#     get_particle = getattr(pop, "get_particle", None)
#     particles_attr = getattr(pop, "particles", None)

#     def _get_dp_m(part) -> float:
#         # Prefer part2pop API
#         if hasattr(part, "get_Ddry"):
#             return float(part.get_Ddry())
#         if hasattr(part, "get_Dcore"):
#             return float(part.get_Dcore())
#         # fallback attribute names
#         for attr in ["Dp", "Dp_m", "diameter_m", "D_m", "D"]:
#             if hasattr(part, attr):
#                 return float(getattr(part, attr))
#         raise AttributeError("Could not infer particle diameter (need get_Ddry/get_Dcore or Dp-like attribute).")

#     for i, pid in enumerate(ids):
#         Ni = float(num_concs[i])
#         if not np.isfinite(Ni) or Ni <= 0:
#             continue

#         if callable(get_particle):
#             part = get_particle(pid)
#         elif particles_attr is not None:
#             part = particles_attr[pid]
#         else:
#             raise AttributeError("Population does not expose get_particle(pid) or particles container.")

#         dp = _get_dp_m(part)  # meters

#         # locate bin: edges[j] <= dp < edges[j+1]
#         j = np.searchsorted(edges_m, dp, side="right") - 1
#         if 0 <= j < nb:
#             N_pop_m3[j] += Ni

#     N_pop_cm3 = N_pop_m3 / 1e6

#     _plot_size_dist_compare(
#         Dp_lo_nm,
#         Dp_hi_nm,
#         N_cm3,
#         N_pop_cm3,
#         outdir / f"{prefix}diag_size_dist.png",
#     )

    # Dp_lo_nm = np.asarray(Dp_lo_nm, dtype=float)
    # Dp_hi_nm = np.asarray(Dp_hi_nm, dtype=float)
    # N_cm3 = np.asarray(N_cm3, dtype=float)

    # Dp_lo_m = Dp_lo_nm * 1e-9
    # Dp_hi_m = Dp_hi_nm * 1e-9

    # dNdlnD_var = build_variable(
    #     "dNdlnD",
    #     "population",
    #     var_cfg={
    #         "D_min": float(np.min(Dp_lo_m)),
    #         "D_max": float(np.max(Dp_hi_m)),
    #         "N_bins": int(len(Dp_lo_m)),
    #         "method": "hist",
    #         "wetsize": False,
    #     },
    # )
    # dNdlnD_pop_m3 = np.asarray(dNdlnD_var.compute(population=pop), dtype=float)

    # dln = np.log(Dp_hi_m / Dp_lo_m)
    # N_pop_m3 = dNdlnD_pop_m3 * dln
    # N_pop_cm3 = N_pop_m3 / 1e6

    # _plot_size_dist_compare(
    #     Dp_lo_nm,
    #     Dp_hi_nm,
    #     N_cm3,
    #     N_pop_cm3,
    #     outdir / f"{prefix}diag_size_dist.png",
    # )

    # species_names = [s.name for s in pop.species]
    # spec_masses = np.asarray(pop.spec_masses, dtype=float)  # (n_particles, n_species)
    # num_concs = np.asarray(pop.num_concs, dtype=float)      # (n_particles,)

    # if spec_masses.ndim != 2 or spec_masses.shape[0] != len(num_concs):
    #     raise ValueError("Unexpected pop.spec_masses / pop.num_concs shapes; cannot compute AMS mass fractions.")

    # mass_kg_m3 = (num_concs[:, None] * spec_masses).sum(axis=0)
    # mass_total = mass_kg_m3.sum()
    # mass_frac_pop = {}
    # if mass_total > 0:
    #     mass_frac_arr = mass_kg_m3 / mass_total
    #     for name, val in zip(species_names, mass_frac_arr):
    #         mass_frac_pop[name] = float(val)

    # keys = ["SO4", "NO3", "OC", "NH4"]
    # obs = {k: float(ams_mass_frac_obs.get(k, 0.0)) for k in keys}
    # pop_vals = {k: float(mass_frac_pop.get(k, np.nan)) for k in keys}

    # _plot_bar_compare(obs, pop_vals, "AMS mass fractions (observed vs reconstructed)", outdir / f"{prefix}diag_ams_mass_fracs.png")

    # pop_tf = {}
    # md = getattr(pop, "metadata", None)
    # if isinstance(md, dict) and "type_fracs" in md:
    #     pop_tf = md["type_fracs"]

    # def _norm(d):
    #     d = {k: float(v) for k, v in d.items() if np.isfinite(v) and float(v) > 0}
    #     s = sum(d.values())
    #     return {k: v / s for k, v in d.items()} if s > 0 else {}

    # obs_tf = _norm(type_fracs_obs)
    # pop_tf = _norm(pop_tf)

    # _plot_bar_compare(obs_tf, pop_tf, "miniSPLAT number fractions (observed vs reconstructed)", outdir / f"{prefix}diag_minisplat_type_fracs.png")

    # summary = outdir / f"{prefix}diagnostics_summary.txt"
    # with summary.open("w") as f:
    #     f.write("Diagnostics summary\n")
    #     f.write("===================\n\n")
    #     f.write(f"Total N (FIMS, cm^-3): {float(np.nansum(N_cm3)):.6g}\n")
    #     f.write(f"Total N (pop,  cm^-3): {float(np.nansum(N_pop_cm3)):.6g}\n\n")
    #     f.write("AMS mass fractions (observed):\n")
    #     for k in keys:
    #         f.write(f"  {k}: {obs.get(k, np.nan):.4f}\n")
    #     f.write("\nAMS mass fractions (reconstructed):\n")
    #     for k in keys:
    #         v = pop_vals.get(k, np.nan)
    #         f.write(f"  {k}: {v if not np.isnan(v) else float('nan'):.4f}\n")
    #     f.write("\nminiSPLAT number fractions (observed):\n")
    #     for k, v in sorted(obs_tf.items()):
    #         f.write(f"  {k}: {v:.4f}\n")
    #     f.write("\nminiSPLAT number fractions (reconstructed / builder):\n")
    #     for k, v in sorted(pop_tf.items()):
    #         f.write(f"  {k}: {v:.4f}\n")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to .../separate_tools/datasets")
    ap.add_argument("--date", required=True, help="YYYYMMDD or YYYY-MM-DD (e.g. 20160425)")
    ap.add_argument("--z", type=float, default=100.0)
    ap.add_argument("--dz", type=float, default=2.0)
    ap.add_argument("--cloud-flag", type=int, default=0)
    ap.add_argument("--max-dp-nm", type=float, default=1000.0)
    ap.add_argument("--splat-cutoff-nm", type=float, default=85.0)
    ap.add_argument("--plots", action="store_true", help="Make diagnostic plots")
    ap.add_argument("--outdir", default=".", help="Where to save plots (and optional CSVs)")
    ap.add_argument("--prefix", default="", help="Prefix for plot filenames")

    args = ap.parse_args()

    datasets_root = Path(args.root).expanduser().resolve()
    date = _parse_date(args.date)

    paths = resolve_hiscale_paths(datasets_root, date)

    print("Resolved files:")
    for k in ["dataset_dir", "fims_file", "aimms_file", "ams_file", "splat_file", "fims_bins_file"]:
        print(f"  {k}: {paths[k]}")

    # --- Observational summaries for diagnostics ---
    Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3 = _read_fims_avg_size_dist(
        fims_file=str(paths["fims_file"]),
        fims_bins_file=str(paths["fims_bins_file"]),
        aimms_file=str(paths["aimms_file"]),
        z=args.z,
        dz=args.dz,
        cloud_flag_value=args.cloud_flag,
        max_dp_nm=args.max_dp_nm,
        region_filter=None,
        aimms_time_col="Time(UTC)",
        aimms_cloud_col=None,
        fims_time_col=None,   # allow the reader to auto-detect the correct time column
        fims_density_measure="ln",  # default: treat FIMS counts as dN/dlnDp unless overridden
    )

    splat_species = {
        "BC": ["soot"],
        "IEPOX": ["IEPOX_SOA"],
        "BB": ["BB", "BB_SOA"],
        "OIN": ["Dust"],
        "AS": ["sulfate_nitrate_org", "nitrate_amine_org"],
    }

    type_fracs_obs, type_fracs_err = _read_minisplat_number_fractions(
        splat_file=str(paths["splat_file"]),
        aimms_file=str(paths["aimms_file"]),
        fims_file=str(paths["fims_file"]),
        splat_species=splat_species,
        z=args.z,
        dz=args.dz,
        cloud_flag_value=args.cloud_flag,
        region_filter=None,
    )

    s = float(sum(type_fracs_obs.values())) if type_fracs_obs else 0.0
    tf = {k: (float(v) / s) for k, v in type_fracs_obs.items()} if s > 0 else {}

    ams_mass_frac, ams_mass_frac_err, measured_mass, measured_mass_err = _read_ams_mass_fractions(
        ams_file=str(paths["ams_file"]),
        aimms_file=str(paths["aimms_file"]),
        fims_file=str(paths["fims_file"]),
        z=args.z,
        dz=args.dz,
        cloud_flag_value=args.cloud_flag,
    )

    type_templates = {
        "BC": {"BC": 1.0},
        "AS": {"SO4": 0.6, "NH4": 0.4},
        "IEPOX": {"OC": 0.8, "SO4": 0.2},
        "OIN": {"OIN": 1.0},
    }

    cfg = {
        "type": "hiscale_observations",
        "fims_file": str(paths["fims_file"]),
        "aimms_file": str(paths["aimms_file"]),
        "splat_file": str(paths["splat_file"]),
        "ams_file": str(paths["ams_file"]),
        "z": args.z,
        "dz": args.dz,
        "splat_species": splat_species,
        "cloud_flag_value": args.cloud_flag,
        "max_dp_nm": args.max_dp_nm,
        "splat_cutoff_nm": args.splat_cutoff_nm,
        "composition_strategy": "templates_fallback_to_ams",
        "type_templates": type_templates,
        "ams_species_map": {"SO4": "SO4", "NO3": "NO3", "OC": "OC", "NH4": "NH4"},
        "D_is_wet": False,
        # builder requires bins file
        "fims_bins_file": str(paths["fims_bins_file"]),
        # allow override of density measure if needed:
        "fims_density_measure": "ln",
    }

    pop = build_population(cfg)

    print("\nBuilt population:")
    print("  n_particles:", len(pop.ids))
    print("  total N [m^-3]:", float(pop.get_Ntot()))
    print("  species:", [s.name for s in pop.species])

    N_fims_m3 = float((np.asarray(N_cm3) * 1e6).sum())
    N_pop_m3 = float(pop.get_Ntot())
    if N_fims_m3 > 0 and N_pop_m3 > 0:
        N_scale = N_fims_m3 / N_pop_m3
        if not np.isclose(N_scale, 1.0, rtol=1e-4, atol=0):
            print(f"Scaling population number concentrations by {N_scale:.6f} to match FIMS total N.")
            pop.num_concs = list(np.asarray(pop.num_concs, dtype=float) * N_scale)
            N_pop_m3 = float(pop.get_Ntot())
            md = getattr(pop, "metadata", None)
            if isinstance(md, dict):
                md["N_scaling_applied"] = N_scale

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / f"{args.prefix}summary.txt", "w") as f:
        f.write(f"N_fims_m3 {N_fims_m3:.6e}\n")
        f.write(f"N_pop_m3  {N_pop_m3:.6e}\n")
        f.write(f"ratio     {N_pop_m3 / N_fims_m3 if N_fims_m3 > 0 else float('nan'):.6f}\n")

    if args.plots:
        make_diagnostics(
            pop=pop,
            Dp_lo_nm=Dp_lo_nm,
            Dp_hi_nm=Dp_hi_nm,
            N_cm3=N_cm3,
            type_fracs_obs=tf,
            ams_mass_frac_obs=ams_mass_frac,
            outdir=str(outdir),
            prefix=(args.prefix or f"{args.date}_"),
        )
        print(f"\nWrote plots + summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()



# """
# Build an initial population from HI-SCALE observations using part2pop's
# hiscale_observations builder, with robust date-driven file selection.

# This script is intentionally *dataset-layout aware* (separate_tools/datasets),
# but keeps all “guessing” in the script layer (not in the part2pop builder).

# Usage:
#   python scripts/build_init_pop_from_obs.py \
#     --root /path/to/multipart_archived/separate_tools/datasets \
#     --date 20160425 --z 100 --dz 2

# Example (your machine):
#   python scripts/build_init_pop_from_obs.py \
#     --root /Users/fier887/Library/CloudStorage/OneDrive-PNNL/Code/multipart_archived/separate_tools/datasets \
#     --date 20160425
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path
# import re
# from datetime import datetime
# from typing import Iterable, List, Dict

# from part2pop.population import build_population
# from part2pop.population.factory.hiscale_observations import (
#     _read_fims_avg_size_dist,
#     _read_minisplat_number_fractions,
#     _read_ams_mass_fractions,
# )

# from part2pop.analysis.builder import build_variable




# def _parse_date(date_str: str) -> datetime:
#     """Accepts 'YYYYMMDD' or 'YYYY-MM-DD'."""
#     date_str = date_str.strip()
#     if re.fullmatch(r"\d{8}", date_str):
#         return datetime.strptime(date_str, "%Y%m%d")
#     if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
#         return datetime.strptime(date_str, "%Y-%m-%d")
#     raise ValueError(f"Unrecognized date format: {date_str}. Use YYYYMMDD or YYYY-MM-DD.")


# def _pick_one(paths: Iterable[Path], label: str) -> Path:
#     paths = sorted([Path(p) for p in paths])
#     if len(paths) == 0:
#         raise FileNotFoundError(f"Could not find any file for {label}.")
#     if len(paths) == 1:
#         return paths[0]
#     print(f"WARNING: multiple candidates for {label}; using {paths[-1].name}")
#     return paths[-1]


# def _extract_dates_from_names(paths: Iterable[Path]) -> List[str]:
#     """
#     Extract YYYYMMDD occurrences from filenames to help with diagnostics.
#     """
#     dates = set()
#     pat = re.compile(r"(20\d{6})")  # catches 20160425 etc.
#     for p in paths:
#         m = pat.search(p.name)
#         if m:
#             dates.add(m.group(1))
#     return sorted(dates)



# def resolve_hiscale_paths(datasets_root: Path, date: datetime) -> Dict[str, Path]:
#     """
#     Resolve files under datasets_root for a given date.

#     We prefer the directory:
#       datasets_root/HISCALE_data_MMDD

#     If that directory does not exist, we search all HISCALE_data_* directories
#     for ANY file containing the YYYYMMDD string for the requested date.

#     Once we choose a dataset_dir, we resolve instrument files using patterns
#     that match what you showed for HISCALE_data_0425:
#       - FIMS:  FIMS_G1_YYYYMMDD_R*_HISCALE_001s.txt
#       - AIMMS: AIMMS20_G1_YYYYMMDD*_*_HISCALE020h.txt
#       - AMS:   HiScaleAMS_G1_YYYYMMDD_R*.txt
#       - SPLAT: Splat_Composition_DD-Mon-YYYY.txt
#     """
#     datasets_root = Path(datasets_root).expanduser().resolve()
#     if not datasets_root.exists():
#         raise FileNotFoundError(f"datasets_root not found: {datasets_root}")

#     mmdd = date.strftime("%m%d")
#     yyyymmdd = date.strftime("%Y%m%d")
#     yyyy = date.strftime("%Y")
#     day = date.strftime("%d").lstrip("0")
#     mon_abbrev = date.strftime("%b")  # Apr, May, ...

#     preferred = datasets_root / f"HISCALE_data_{mmdd}"

#     # 1) pick dataset_dir
#     if preferred.exists() and preferred.is_dir():
#         dataset_dir = preferred
#     else:
#         # Search all HISCALE_data_* directories for any file containing the date string
#         candidates = []
#         for d in sorted(datasets_root.glob("HISCALE_data_*")):
#             if not d.is_dir():
#                 continue
#             if any(d.glob(f"*{yyyymmdd}*")):
#                 candidates.append(d)

#         if len(candidates) == 0:
#             # Provide diagnostic: what dates are present?
#             all_files = list(datasets_root.glob("HISCALE_data_*/*"))
#             dates_present = _extract_dates_from_names(all_files)
#             msg = (
#                 f"No HISCALE_data_* directory under {datasets_root} contains any file with date '{yyyymmdd}'.\n"
#                 f"Dates present (from filenames, sample): {dates_present[:20]}"
#                 + (" ..." if len(dates_present) > 20 else "")
#             )
#             raise FileNotFoundError(msg)

#         if len(candidates) > 1:
#             print("WARNING: multiple dataset dirs contain the requested date; using:", candidates[-1].name)
#         dataset_dir = candidates[-1]

#     # 2) resolve instrument files within dataset_dir using the patterns that match your 0425 folder
#     fims_candidates = list(dataset_dir.glob(f"FIMS_G1_{yyyymmdd}_R*_HISCALE_001s.txt"))
#     aimms_candidates = list(dataset_dir.glob(f"AIMMS20_G1_{yyyymmdd}*_HISCALE020h.txt"))
#     ams_candidates = list(dataset_dir.glob(f"HiScaleAMS_G1_{yyyymmdd}_R*.txt"))
#     splat_candidates = list(dataset_dir.glob(f"Splat_Composition_{day}-{mon_abbrev}-{yyyy}.txt"))

#     # If strict patterns fail, try slightly broader patterns before giving up
#     if len(fims_candidates) == 0:
#         fims_candidates = list(dataset_dir.glob(f"FIMS*_G1*{yyyymmdd}*HISCALE*001s*.txt"))
#     if len(aimms_candidates) == 0:
#         aimms_candidates = list(dataset_dir.glob(f"AIMMS*G1*{yyyymmdd}*HISCALE020h*.txt"))
#     if len(ams_candidates) == 0:
#         ams_candidates = list(dataset_dir.glob(f"HiScaleAMS*G1*{yyyymmdd}*R*.txt"))
#     if len(splat_candidates) == 0:
#         # fallback: any Splat_Composition file in the folder that contains YYYY (still date-specific folder usually)
#         splat_candidates = list(dataset_dir.glob("Splat_Composition_*.txt"))

#     # If we still can't find required files, print what's in the directory to diagnose fast.
#     missing = []
#     if len(fims_candidates) == 0:
#         missing.append("FIMS")
#     if len(aimms_candidates) == 0:
#         missing.append("AIMMS")
#     if len(ams_candidates) == 0:
#         missing.append("AMS")
#     if len(splat_candidates) == 0:
#         missing.append("SPLAT")

#     if missing:
#         listing = sorted([p.name for p in dataset_dir.iterdir() if p.is_file()])
#         raise FileNotFoundError(
#             f"Missing required files in {dataset_dir} for date {yyyymmdd}: {missing}\n"
#             f"Directory listing:\n  " + "\n  ".join(listing[:80]) + ("\n  ..." if len(listing) > 80 else "")
#         )

#     fims_file = _pick_one(fims_candidates, "FIMS")
#     aimms_file = _pick_one(aimms_candidates, "AIMMS")
#     ams_file = _pick_one(ams_candidates, "AMS")
#     splat_file = _pick_one(splat_candidates, "miniSPLAT composition")

#     bins_candidates = list(dataset_dir.glob("HISCALE_FIMS_bins_R*.txt"))
#     bins_file = _pick_one(bins_candidates, "FIMS bins")

#     return {
#         "dataset_dir": dataset_dir,
#         "fims_file": fims_file,
#         "aimms_file": aimms_file,
#         "ams_file": ams_file,
#         "splat_file": splat_file,
#         "fims_bins_file": bins_file,
#     }

# def _iter_pop_particles(pop):
#     """
#     Yield tuples: (particle_obj, N_m3)
#     Attempts a few common storage patterns used by particle population classes.
#     """
#     # Most likely in your case: pop.ids + pop.num_concs + pop.get_particle(id) OR pop.particles dict
#     if hasattr(pop, "ids") and hasattr(pop, "num_concs"):
#         ids = list(pop.ids)
#         Ns = list(pop.num_concs)
#         # Try get_particle
#         if hasattr(pop, "get_particle"):
#             for pid, N in zip(ids, Ns):
#                 yield pop.get_particle(pid), float(N)
#             return
#         # Try particles dict
#         if hasattr(pop, "particles") and isinstance(pop.particles, dict):
#             for pid, N in zip(ids, Ns):
#                 yield pop.particles[pid], float(N)
#             return

#     # Fallback: pop.particles list-like
#     if hasattr(pop, "particles"):
#         parts = pop.particles
#         if isinstance(parts, dict):
#             for pid, p in parts.items():
#                 # no N info; assume separate array exists (handled above). Can't proceed.
#                 continue
#         else:
#             # no N info; can't proceed
#             return

#     raise AttributeError("Could not iterate particles + number concentrations from population object.")


# def _get_particle_diameter_m(p):
#     for attr in ["Dp", "Dp_m", "diameter_m", "D_m"]:
#         if hasattr(p, attr):
#             return float(getattr(p, attr))
#     # some objects store as property/method
#     for fn in ["get_Dp", "get_diameter_m"]:
#         if hasattr(p, fn):
#             return float(getattr(p, fn)())
#     raise AttributeError("Particle diameter not found on particle object.")


# def _get_particle_species_masses(p):
#     """
#     Return 1D numpy array of species masses aligned to pop.species ordering.
#     """
#     for attr in ["spec_masses", "species_masses", "masses"]:
#         if hasattr(p, attr):
#             m = getattr(p, attr)
#             return np.array(m, dtype="float64").copy()
#     if hasattr(p, "get_spec_masses"):
#         return np.array(p.get_spec_masses(), dtype="float64").copy()
#     raise AttributeError("Particle species masses not found on particle object.")


# def _plot_size_dist_compare(Dp_lo_nm, Dp_hi_nm, N_fims_cm3, N_pop_cm3, outpath):
#     import matplotlib.pyplot as plt

#     Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)
    
    
#     plt.figure()
#     plt.plot(Dp_mid_nm, N_fims_cm3, marker="o", linestyle="-", label="FIMS (avg, filtered)")
#     plt.plot(Dp_mid_nm, N_pop_cm3, marker="s", linestyle="--", label="Reconstructed pop")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel("Dp (nm)")
#     plt.ylabel("N(Dp) (cm$^{-3}$ per bin)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=200)
#     plt.close()


# def _plot_bar_compare(d_true: dict, d_pop: dict, title: str, outpath: str):
#     import matplotlib.pyplot as plt

#     keys = sorted(set(d_true.keys()) | set(d_pop.keys()))
#     true = np.array([d_true.get(k, 0.0) for k in keys], dtype="float64")
#     pop  = np.array([d_pop.get(k, 0.0) for k in keys], dtype="float64")

#     x = np.arange(len(keys))
#     w = 0.40

#     plt.figure()
#     plt.bar(x - w/2, true, width=w, label="Observed")
#     plt.bar(x + w/2, pop,  width=w, label="Reconstructed")
#     plt.xticks(x, keys, rotation=45, ha="right")
#     plt.ylabel("Fraction")
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=200)
#     plt.close()

# def make_diagnostics(
#     *,
#     pop,
#     Dp_lo_nm,
#     Dp_hi_nm,
#     N_cm3,
#     type_fracs_obs,
#     ams_mass_frac_obs,
#     outdir=".",
#     prefix="",
# ):
#     """
#     Diagnostics:
#       1) FIMS measured size distribution vs reconstructed from pop
#       2) AMS mass fractions: observed vs reconstructed
#       3) miniSPLAT number fractions: observed vs reconstructed

#     Assumptions:
#       - pop.num_concs are in m^-3
#       - pop particles have diameter accessible as particle.Dp (meters). If not, we try particle.D (meters).
#       - pop.spec_masses are per-particle species masses in kg aligned to pop.species ordering.
#     """
#     from pathlib import Path
#     import numpy as np
#     import matplotlib.pyplot as plt

#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     # -----------------------
#     # Helper: pull particle diameter in meters
#     # -----------------------

#     def _get_dp_m(part, *, use_wet: bool = False) -> float:
#         """
#         Return particle diameter [m] for plotting.
#         Uses part2pop Particle API (get_Ddry/get_Dwet).
#         """
#         if use_wet and hasattr(part, "get_Dwet"):
#             return float(part.get_Dwet())
#         if hasattr(part, "get_Ddry"):
#             return float(part.get_Ddry())
#         if hasattr(part, "get_Dcore"):
#             return float(part.get_Dcore())
#         raise AttributeError(
#             "Could not infer particle diameter. Expected get_Ddry/get_Dwet/get_Dcore."
#         )

#     # FIXME: remove:
#     # def _get_dp_m(p, *, wet: bool = False) -> float:
#     #     """
#     #     Return particle diameter [m]. Uses part2pop.aerosol_particle.Particle API.
#     #     If wet=True, use Dwet; otherwise use Ddry.
#     #     """
#     #     if wet and hasattr(p, "get_Dwet"):
#     #         return float(p.get_Dwet())
#     #     if hasattr(p, "get_Ddry"):
#     #         return float(p.get_Ddry())

#     #     # fallback(s)
#     #     if hasattr(p, "get_Dcore"):
#     #         return float(p.get_Dcore())

#     #     raise AttributeError(
#     #         "Particle does not provide get_Ddry/get_Dwet/get_Dcore; cannot infer diameter."
#     #     )

#     # FIXME: remove when done
#     # def _get_dp_m(p):
#     #     """
#     #     Try common diameter/radius attribute names used in particle classes.
#     #     Returns diameter in meters.
#     #     """
#     #     # Direct diameter fields (meters)
#     #     for attr in [
#     #         "Dp", "D", "diameter", "diameter_m", "Dp_m",
#     #         "Ddry", "D_dry", "Dp_dry",
#     #         "Dwet", "D_wet", "Dp_wet",
#     #     ]:
#     #         if hasattr(p, attr):
#     #             val = getattr(p, attr)
#     #             try:
#     #                 return float(val)
#     #             except Exception:
#     #                 pass

#     #     # Methods (if particle stores diameter behind a getter)
#     #     for meth in ["get_Dp", "get_diameter", "get_diameter_m", "Dp_m", "diameter_m"]:
#     #         if hasattr(p, meth) and callable(getattr(p, meth)):
#     #             try:
#     #                 return float(getattr(p, meth)())
#     #             except Exception:
#     #                 pass

#     #     # Radius fields (meters) -> convert to diameter
#     #     for attr in ["r", "radius", "radius_m", "r_m", "r_dry", "r_wet"]:
#     #         if hasattr(p, attr):
#     #             val = getattr(p, attr)
#     #             try:
#     #                 return 2.0 * float(val)
#     #             except Exception:
#     #                 pass

#     #     # If we get here, print something useful once
#     #     keys = [k for k in dir(p) if not k.startswith("_")]
#     #     raise AttributeError(
#     #         "Could not infer particle diameter. "
#     #         "Tried common diameter/radius fields and getter methods.\n"
#     #         f"Available attributes (sample): {keys[:60]}"
#     #     )

#     # FIXME: remove when done
#     # def _get_dp_m(p):
#     #     if hasattr(p, "Dp"):
#     #         return float(p.Dp)
#     #     if hasattr(p, "D"):
#     #         return float(p.D)
#     #     raise AttributeError("Particle object has no attribute 'Dp' or 'D' for diameter (meters).")
#     def _pop_ams_mass_fracs(pop, ams_species=("SO4", "NO3", "OC", "NH4")) -> dict:
#         """
#         Compute AMS-like bulk mass fractions from the constructed population.
#         Uses DRY mass and includes only AMS species.
#         Returns dict species->fraction (sums to 1 over available AMS species).
#         """
#         masses = {s: 0.0 for s in ams_species}

#         for pid in pop.ids:
#             part = pop.get_particle(pid) if hasattr(pop, "get_particle") else pop.particles[pid]
#             Ni = float(pop.num_concs[pop.ids.index(pid)]) if isinstance(pop.ids, list) else None  # fallback below

#             # Safer: use index mapping once if you have it; otherwise:
#             # If pop has dict-like storage, adapt accordingly.
#             # Many part2pop populations store num_concs aligned with ids:
#             try:
#                 idx = list(pop.ids).index(pid)
#                 Ni = float(pop.num_concs[idx])
#             except Exception:
#                 pass
#             if Ni is None:
#                 continue

#             for s in ams_species:
#                 try:
#                     mi = float(part.get_spec_mass(s))  # kg per particle
#                 except Exception:
#                     mi = 0.0
#                 masses[s] += Ni * mi  # kg m^-3

#         tot = sum(masses.values())
#         if tot <= 0:
#             return {s: 0.0 for s in ams_species}
#         return {s: masses[s] / tot for s in ams_species}
    
#         # -----------------------
#         # 1) Size distribution: measured (FIMS) vs reconstructed (population)
#         # -----------------------
#         Dp_lo_nm = np.asarray(Dp_lo_nm, dtype=float)
#         Dp_hi_nm = np.asarray(Dp_hi_nm, dtype=float)
#         N_cm3 = np.asarray(N_cm3, dtype=float)

#         Dp_lo_m = Dp_lo_nm * 1e-9
#         Dp_hi_m = Dp_hi_nm * 1e-9

#         # Build dNdlnD on the SAME binning as FIMS
#         # (method='hist' gives you exact binning; KDE will smear / change totals)
#         dNdlnD_var = build_variable(
#             "dNdlnD",
#             "population",
#             var_cfg={
#                 "D_min": float(np.min(Dp_lo_m)),
#                 "D_max": float(np.max(Dp_hi_m)),
#                 "N_bins": int(len(Dp_lo_m)),   # bins count
#                 "method": "hist",              # <-- use hist for exact conservation
#                 "wetsize": False,
#             },
#         )

#         dNdlnD_pop_m3 = np.asarray(dNdlnD_var.compute(population=pop), dtype=float)

#         # Convert dN/dlnD to per-bin N by multiplying by dlnD for each bin
#         dlnD = np.log(Dp_hi_m / Dp_lo_m)
#         N_pop_m3 = dNdlnD_pop_m3 * dlnD
#         N_pop_cm3 = N_pop_m3 / 1e6

#         Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)

#         plt.figure()
#         plt.plot(Dp_mid_nm, N_cm3, marker="o", linestyle="-", label="FIMS (observed, per bin)")
#         plt.plot(Dp_mid_nm, N_pop_cm3, marker="s", linestyle="--", label="Population (from dNdlnD → per bin)")
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.xlabel("Dp (nm)")
#         plt.ylabel("N (cm$^{-3}$ per bin)")
#         plt.title("Size distribution: FIMS vs reconstructed population")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(outdir / f"{prefix}diag_size_dist.png", dpi=200)
#         plt.close()

#     # # -----------------------
#     # # 1) Size distribution: measured (FIMS) vs reconstructed (population)
#     # # -----------------------
#     # Dp_lo_nm = np.asarray(Dp_lo_nm, dtype=float)
#     # Dp_hi_nm = np.asarray(Dp_hi_nm, dtype=float)
#     # N_cm3 = np.asarray(N_cm3, dtype=float)

#     # Dp_lo_m = Dp_lo_nm * 1e-9
#     # Dp_hi_m = Dp_hi_nm * 1e-9

#     # # Build dNdlnD on the SAME binning as FIMS
#     # # (method='hist' gives you exact binning; KDE will smear / change totals)
#     # dNdlnD_var = build_variable(
#     #     "dNdlnD",
#     #     "population",
#     #     var_cfg={
#     #         "D_min": float(np.min(Dp_lo_m)),
#     #         "D_max": float(np.max(Dp_hi_m)),
#     #         "N_bins": int(len(Dp_lo_m)),   # bins count
#     #         "method": "hist",              # <-- use hist for exact conservation
#     #         "wetsize": False,
#     #     },
#     # )

#     # dNdlnD_pop_m3 = np.asarray(dNdlnD_var.compute(population=pop), dtype=float)

#     # # Convert dN/dlnD to per-bin N by multiplying by dlnD for each bin
#     # dlnD = np.log(Dp_hi_m / Dp_lo_m)
#     # N_pop_m3 = dNdlnD_pop_m3 * dlnD
#     # N_pop_cm3 = N_pop_m3 / 1e6

#     # Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)

#     # plt.figure()
#     # plt.plot(Dp_mid_nm, N_cm3, marker="o", linestyle="-", label="FIMS (observed, per bin)")
#     # plt.plot(Dp_mid_nm, N_pop_cm3, marker="s", linestyle="--", label="Population (from dNdlnD → per bin)")
#     # plt.xscale("log")
#     # plt.yscale("log")
#     # plt.xlabel("Dp (nm)")
#     # plt.ylabel("N (cm$^{-3}$ per bin)")
#     # plt.title("Size distribution: FIMS vs reconstructed population")
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.savefig(outdir / f"{prefix}diag_size_dist.png", dpi=200)
#     # plt.close()

#     # # -----------------------
#     # # 1) Size distribution: measured (FIMS) vs reconstructed (population)
#     # # -----------------------
#     # Dp_lo_nm = np.asarray(Dp_lo_nm, dtype=float)
#     # Dp_hi_nm = np.asarray(Dp_hi_nm, dtype=float)
#     # N_cm3 = np.asarray(N_cm3, dtype=float)

#     # edges_nm = np.concatenate([Dp_lo_nm[:1], Dp_hi_nm])  # len = nbins+1
#     # nb = len(Dp_lo_nm)

#     # # Reconstruct N in each FIMS bin from population
#     # # N_pop_m3 = np.zeros(nb, dtype=float)

#     # # pop.ids are IDs; assume pop.get_particle(id) exists; else pop.particles dict/list
#     # # We'll support two common patterns:
#     # #   - pop.get_particle(pid)
#     # #   - pop.particles[pid]
#     # get_particle = getattr(pop, "get_particle", None)
#     # particles_attr = getattr(pop, "particles", None)

#     # for i, pid in enumerate(pop.ids):
#     #     Ni_m3 = float(pop.num_concs[i])  # already SI
#     #     if Ni_m3 <= 0 or not np.isfinite(Ni_m3):
#     #         continue

#     #     if callable(get_particle):
#     #         part = get_particle(pid)
#     #     elif particles_attr is not None:
#     #         part = particles_attr[pid]
#     #     else:
#     #         raise AttributeError("Population does not expose get_particle(pid) or particles container.")

#     #     dp_nm = _get_dp_m(part, use_wet=False) * 1e9
#     #     dNdlnD_var = build_variable(
#     #         'dNdlnD', "population", 
#     #         var_cfg={'D_min':min(Dp_lo_nm)*1e-9,'D_max':max(Dp_hi_nm)*1e-9,"N_bins":len(Dp_lo_nm)-1,'method':'kde', 'wetsize':False})
        
#     #     N_pop_cm3 = dNdlnD_var.compute(population=pop)*1e-6
#     #     # # find bin index
#     #     # j = np.searchsorted(edges_nm, dp_nm, side="right") - 1
#     #     # if 0 <= j < nb:
#     #     #     N_pop_m3[j] += Ni_m3

#     # N_pop_cm3 = N_pop_m3 / 1e6  # m^-3 -> cm^-3
#     # Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)

#     # plt.figure()
#     # plt.plot(Dp_mid_nm, N_cm3, marker="o", linestyle="-", label="FIMS (observed)")
#     # plt.plot(Dp_mid_nm, N_pop_cm3, marker="s", linestyle="--", label="Population (reconstructed)")
#     # plt.xscale("log")
#     # plt.yscale("log")
#     # plt.xlabel("Dp (nm)")
#     # plt.ylabel("N(Dp) (cm$^{-3}$)")
#     # plt.title("Size distribution: FIMS vs reconstructed population")
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.savefig(outdir / f"{prefix}diag_size_dist.png", dpi=200)
#     # plt.close()

#     # -----------------------
#     # 2) AMS mass fractions: observed vs reconstructed
#     # -----------------------
#     # Reconstruct bulk mass fractions from population:
#     # total mass per species = sum_i Ni * m_i(spec)
#     species_names = [s.name for s in pop.species]
#     n_spec = len(species_names)

#     # pop.spec_masses: list/array of per-particle vectors aligned to species ordering
#     # Some implementations store as list of arrays; some as 2D array.
#     spec_masses = np.asarray(pop.spec_masses, dtype=float)  # shape (n_particles, n_spec)
#     num_concs = np.asarray(pop.num_concs, dtype=float)      # shape (n_particles,)

#     if spec_masses.ndim != 2 or spec_masses.shape[1] != n_spec:
#         raise ValueError(f"Unexpected pop.spec_masses shape {spec_masses.shape}; expected (n_particles, {n_spec})")

#     # total mass per species in kg/m^3
#     mass_kg_m3 = (num_concs[:, None] * spec_masses).sum(axis=0)
#     mass_total = mass_kg_m3.sum()
#     if mass_total > 0:
#         mass_frac_pop = mass_kg_m3 / mass_total
#     else:
#         mass_frac_pop = np.zeros_like(mass_kg_m3)

#     # Compare only the AMS keys you care about (typically SO4, NO3, OC, NH4)
#     keys = ["SO4", "NO3", "OC", "NH4"]
#     obs = [float(ams_mass_frac_obs.get(k, np.nan)) for k in keys]

#     # Map to population species by exact name match
#     pop_vals = []
#     for k in keys:
#         if k in species_names:
#             pop_vals.append(float(mass_frac_pop[species_names.index(k)]))
#         else:
#             pop_vals.append(np.nan)

#     x = np.arange(len(keys))
#     width = 0.4

#     plt.figure()
#     plt.bar(x - width/2, obs, width, label="AMS (observed)")
#     plt.bar(x + width/2, pop_vals, width, label="Population (reconstructed)")
#     plt.xticks(x, keys)
#     plt.ylabel("Mass fraction")
#     plt.ylim(0, 1)
#     plt.title("AMS bulk mass fractions: observed vs reconstructed")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outdir / f"{prefix}diag_ams_mass_fracs.png", dpi=200)
#     plt.close()

#     # -----------------------
#     # 3) miniSPLAT number fractions: observed vs reconstructed
#     # -----------------------
#     # For reconstructed type fractions, use pop.metadata["type_fracs"] if present (best),
#     # else approximate by splitting particles by a stored ptype label if you have it.
#     pop_tf = None
#     md = getattr(pop, "metadata", None)
#     if isinstance(md, dict) and "type_fracs" in md:
#         pop_tf = md["type_fracs"]

#     # fallback: if metadata isn't there, we can't infer type unless particle stores it
#     if pop_tf is None:
#         pop_tf = {}

#     # normalize both
#     def _norm(d):
#         d = {k: float(v) for k, v in d.items() if np.isfinite(v) and float(v) > 0}
#         s = sum(d.values())
#         return {k: v/s for k, v in d.items()} if s > 0 else {}

#     obs_tf = _norm(type_fracs_obs)
#     pop_tf = _norm(pop_tf)

#     # union of keys for plotting
#     labels = sorted(set(obs_tf.keys()) | set(pop_tf.keys()))
#     obs_vals = [obs_tf.get(k, 0.0) for k in labels]
#     pop_vals = [pop_tf.get(k, 0.0) for k in labels]

#     x = np.arange(len(labels))
#     plt.figure(figsize=(max(6, 0.8*len(labels)), 4))
#     plt.bar(x - width/2, obs_vals, width, label="miniSPLAT (observed)")
#     plt.bar(x + width/2, pop_vals, width, label="Population (builder tf)")
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.ylabel("Number fraction")
#     plt.ylim(0, 1)
#     plt.title("miniSPLAT number fractions: observed vs reconstructed")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outdir / f"{prefix}diag_minisplat_type_fracs.png", dpi=200)
#     plt.close()



















# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True, help="Path to .../separate_tools/datasets")
#     ap.add_argument("--date", required=True, help="YYYYMMDD or YYYY-MM-DD (e.g. 20160425)")
#     ap.add_argument("--z", type=float, default=100.0)
#     ap.add_argument("--dz", type=float, default=2.0)
#     ap.add_argument("--cloud-flag", type=int, default=0)
#     ap.add_argument("--max-dp-nm", type=float, default=1000.0)
#     ap.add_argument("--splat-cutoff-nm", type=float, default=85.0)
#     ap.add_argument("--plots", action="store_true", help="Make diagnostic plots")
#     ap.add_argument("--outdir", default=".", help="Where to save plots (and optional CSVs)")
#     ap.add_argument("--prefix", default="", help="Prefix for plot filenames")

#     args = ap.parse_args()

#     datasets_root = Path(args.root).expanduser().resolve()
#     date = _parse_date(args.date)

#     paths = resolve_hiscale_paths(datasets_root, date)

#     print("Resolved files:")
#     for k in ["dataset_dir", "fims_file", "aimms_file", "ams_file", "splat_file", "fims_bins_file"]:
#         print(f"  {k}: {paths[k]}")

#     # --- Observational summaries for diagnostics ---
#     # IMPORTANT: do NOT pass fims_bins_file here (your installed function doesn't accept it)
#     Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3 = _read_fims_avg_size_dist(
#         fims_file=str(paths["fims_file"]),
#         aimms_file=str(paths["aimms_file"]),
#         z=args.z,
#         dz=args.dz,
#         cloud_flag_value=args.cloud_flag,
#         # max_dp_nm=args.max_dp_nm,
#         region_filter=None,
#         aimms_time_col="Time(UTC)",
#         aimms_cloud_col=None,
#         fims_time_col="Start_UTC",   # <-- use the FIMS time column name your reader provides
#     )

#     splat_species = {
#         "BC": ["soot"],
#         "IEPOX": ["IEPOX_SOA"],
#         "BB": ["BB", "BB_SOA"],
#         "OIN": ["Dust"],
#         "AS": ["sulfate_nitrate_org", "nitrate_amine_org"],
#     }

#     type_fracs_obs, type_fracs_err = _read_minisplat_number_fractions(
#         splat_file=str(paths["splat_file"]),
#         aimms_file=str(paths["aimms_file"]),
#         fims_file=str(paths["fims_file"]),
#         splat_species=splat_species,
#         z=args.z,
#         dz=args.dz,
#         cloud_flag_value=args.cloud_flag,
#         region_filter=None,
#     )

#     s = float(sum(type_fracs_obs.values()))
#     tf = {k: (float(v) / s) for k, v in type_fracs_obs.items()} if s > 0 else {}

#     ams_mass_frac, ams_mass_frac_err, measured_mass, measured_mass_err = _read_ams_mass_fractions(
#         ams_file=str(paths["ams_file"]),
#         aimms_file=str(paths["aimms_file"]),
#         fims_file=str(paths["fims_file"]),
#         z=args.z,
#         dz=args.dz,
#         cloud_flag_value=args.cloud_flag,
#     )

#     type_templates = {
#         "BC": {"BC": 1.0},
#         "AS": {"SO4": 0.6, "NH4": 0.4},
#         "IEPOX": {"OC": 0.8, "SO4": 0.2},
#         "OIN": {"OIN": 1.0},
#     }

#     cfg = {
#         "type": "hiscale_observations",
#         "fims_file": str(paths["fims_file"]),
#         "aimms_file": str(paths["aimms_file"]),
#         "splat_file": str(paths["splat_file"]),
#         "ams_file": str(paths["ams_file"]),
#         "z": args.z,
#         "dz": args.dz,
#         "splat_species": splat_species,
#         "cloud_flag_value": args.cloud_flag,
#         "max_dp_nm": args.max_dp_nm,
#         "splat_cutoff_nm": args.splat_cutoff_nm,
#         "composition_strategy": "templates_fallback_to_ams",
#         "type_templates": type_templates,
#         "ams_species_map": {"SO4": "SO4", "NO3": "NO3", "OC": "OC", "NH4": "NH4"},
#         "D_is_wet": False,
#         # KEEP this here; the builder can use it
#         "fims_bins_file": str(paths["fims_bins_file"]),
#     }

#     pop = build_population(cfg)

#     print("\nBuilt population:")
#     print("  n_particles:", len(pop.ids))
#     print("  total N [m^-3]:", float(pop.get_Ntot()))
#     print("  species:", [s.name for s in pop.species])

#     N_fims_m3 = float((np.asarray(N_cm3) * 1e6).sum())
#     N_pop_m3 = float(pop.get_Ntot())

#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     with open(outdir / f"{args.prefix}summary.txt", "w") as f:
#         f.write(f"N_fims_m3 {N_fims_m3:.6e}\n")
#         f.write(f"N_pop_m3  {N_pop_m3:.6e}\n")
#         f.write(f"ratio     {N_pop_m3/N_fims_m3:.6f}\n")

#     if args.plots:
#         make_diagnostics(
#             pop=pop,
#             Dp_lo_nm=Dp_lo_nm,
#             Dp_hi_nm=Dp_hi_nm,
#             N_cm3=N_cm3,
#             type_fracs_obs=tf,
#             ams_mass_frac_obs=ams_mass_frac,
#             outdir=str(outdir),
#             prefix=(args.prefix or f"{args.date}_"),
#         )
#         print(f"\nWrote plots + summary to: {outdir.resolve()}")





































    














































if __name__ == "__main__":
    main()
