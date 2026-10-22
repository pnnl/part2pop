#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a binned lognormal population.

Modes require finite N >= 0, GMD > 0 and GSD > 1. GSD == 1 is a
point mass and is not supported by this continuous-PDF binning factory.
For positive N, sampled bin weights must have finite positive total support.
As before, the represented bins are renormalized to the full configured N;
this does not preserve the fraction of a mode outside a truncated grid.
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from ..utils import expand_compounds_for_population, normalize_population_config
from part2pop import make_particle
from part2pop.species.registry import get_species
from part2pop.species.resolution import resolve_species_name_rows
from .registry import register
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

@register("binned_lognormals")
def build(config):
    # Preserve N_bins before normalize_population_config(), whose historical
    # integer conversion would otherwise turn a value such as 2.5 into 2
    # before this factory can reject it.
    raw_n_bins_config = config.get("N_bins", 100)
    # This factory validates bin counts below. Do not let the generic
    # normalizer coerce or reject them before that diagnostic can run.
    config = dict(config)
    config.pop("N_bins", None)
    config = normalize_population_config(config)
    # fixme: make this +/- certain number of sigmas (rather than min/max diams)
    # D_min = float(config['D_min'])
    # D_max = float(config['D_max'])
    
    N_list = [float(one_N) for one_N in config['N']]
    GMD_list = config['GMD']
    GSD_list = config['GSD']
    
    if isinstance(raw_n_bins_config, (str, bytes)) or np.isscalar(raw_n_bins_config):
        N_bins_list = [raw_n_bins_config] * len(GMD_list)
    else:
        try:
            N_bins_list = list(raw_n_bins_config)
        except TypeError as exc:
            raise ValueError(
                f"N_bins must be an integer >= 2 or an iterable of such values; "
                f"got {raw_n_bins_config!r}."
            ) from exc

    # todo: right now, N_sigmas same for all modes; could be per-mode if needed
    
    N_sigmas = float(config.get('N_sigmas', 5))  # used to set bin ranges for each mode
    
    # If the user provides global D_min/D_max, use them for all modes. Otherwise compute per-mode edges.
    global_D_min = config.get('D_min', None)
    global_D_max = config.get('D_max', None)
    
    if (global_D_min is not None) ^ (global_D_max is not None):
        raise ValueError("Provide both D_min and D_max, or neither.")
    if global_D_min is not None:
        try:
            global_D_min = float(global_D_min)
            global_D_max = float(global_D_max)
        except Exception:
            raise ValueError("D_min/D_max must be numeric (meters).")
        if not (np.isfinite(global_D_min) and np.isfinite(global_D_max)
                and 0 < global_D_min < global_D_max):
            raise ValueError("D_min and D_max must be positive and D_min < D_max; both must be finite.")
    elif not np.isfinite(N_sigmas) or N_sigmas <= 0:
        raise ValueError("N_sigmas must be finite and positive for automatic mode edges.")
    
    aero_spec_names_list = resolve_species_name_rows(config['aero_spec_names'])
    aero_spec_fracs_list = config['aero_spec_fracs']

    # All of these fields describe one value/row per aerosol mode. The
    # construction loop below uses zip(), which silently truncates to the
    # shortest iterable unless strict=True, so reject inconsistent
    # configurations explicitly and retain strict zips as defense in depth.
    mode_lengths = {
        "N": len(N_list),
        "GMD": len(GMD_list),
        "GSD": len(GSD_list),
        "N_bins": len(N_bins_list),
        "aero_spec_names": len(aero_spec_names_list),
        "aero_spec_fracs": len(aero_spec_fracs_list),
    }
    if len(set(mode_lengths.values())) != 1:
        lengths = ", ".join(
            f"{name}={length}" for name, length in mode_lengths.items()
        )
        raise ValueError(
            "Per-mode configuration fields must have the same number of modes; "
            f"got {lengths}."
        )

    for mode_idx, (number, diameter, spread) in enumerate(
            zip(N_list, GMD_list, GSD_list, strict=True)):
        for name, value, valid, constraint in (
                ("N", number, number >= 0, ">= 0"),
                ("GMD", diameter, diameter > 0, "> 0"),
                ("GSD", spread, spread > 1, "> 1")):
            if not np.isfinite(value) or not valid:
                raise ValueError(
                    f"{name} for mode {mode_idx} must be finite and {constraint}; "
                    f"got {value!r}."
                )

    validated_n_bins = []
    for mode_idx, raw_n_bins in enumerate(N_bins_list):
        try:
            n_bins_value = float(raw_n_bins)
            n_bins = int(n_bins_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"N_bins for mode {mode_idx} must be an integer >= 2; "
                f"got {raw_n_bins!r}."
            ) from exc
        if not np.isfinite(n_bins_value) or n_bins_value != n_bins or n_bins < 2:
            raise ValueError(
                f"N_bins for mode {mode_idx} must be an integer >= 2; "
                f"got {raw_n_bins!r}."
            )
        validated_n_bins.append(n_bins)
    N_bins_list = validated_n_bins

    for mode_idx, (mode_names, mode_fracs) in enumerate(
            zip(aero_spec_names_list, aero_spec_fracs_list, strict=True)):
        if len(mode_names) != len(mode_fracs):
            raise ValueError(
                "aero_spec_names and aero_spec_fracs must have matching lengths "
                f"within each mode; mode {mode_idx} has {len(mode_names)} names "
                f"and {len(mode_fracs)} fractions."
            )

        canonical_keys = [str(name).casefold() for name in mode_names]
        if len(canonical_keys) != len(set(canonical_keys)):
            raise ValueError(
                "aero_spec_names contains duplicate canonical species after "
                f"alias resolution in mode {mode_idx}: {mode_names!r}."
            )
    
    # Support compound-like species names (e.g., NaCl, (NH4)2SO4)
    # aero_spec_names_list, aero_spec_fracs_list = expand_compounds_for_population(
    #     aero_spec_names_list, aero_spec_fracs_list
    # )
    species_modifications = config.get('species_modifications', {})
    surface_tension = config.get('surface_tension', 0.072)
    D_is_wet = config.get('D_is_wet', False)
    specdata_path = config.get('specdata_path', None)

    # Build master species list for the *population*, preserving order
    pop_species_names = []
    for mode_names in aero_spec_names_list:
        for name in mode_names:
            if name not in pop_species_names:
                pop_species_names.append(name)
    # Build species objects
    pop_species_list = tuple(
        get_species(spec_name, specdata_path, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    # Create the population object with the right species list
    lognormals_population = ParticlePopulation(
        species=pop_species_list,
        spec_masses=np.empty((0, len(pop_species_list)), dtype=float),
        num_concs=np.empty(0, dtype=float), ids=[],
        species_modifications=species_modifications
    )

    
    part_id = 0
    for mode_idx, (Ntot, GMD, GSD, mode_spec_names, mode_spec_fracs, N_bins) in enumerate(
            zip(
                N_list,
                GMD_list,
                GSD_list,
                aero_spec_names_list,
                aero_spec_fracs_list,
                N_bins_list,
                strict=True,
            )):
        # determine bin edges: either global or per-mode
        if global_D_min is not None:
            # use global edges (N_bins bins => N_bins+1 edges)
            D_edges = np.logspace(np.log10(global_D_min), np.log10(global_D_max), num=int(N_bins) + 1)
        else:
            # per-mode edges centered on GMD spanning N_sigmas in log-space
            mode_D_min = np.exp(np.log(GMD) - 0.5 * N_sigmas * np.log(GSD))
            mode_D_max = np.exp(np.log(GMD) + 0.5 * N_sigmas * np.log(GSD))
            if not (np.isfinite(mode_D_min) and np.isfinite(mode_D_max)
                    and 0 < mode_D_min < mode_D_max):
                raise ValueError(f"Invalid mode edges for mode {mode_idx}: {mode_D_min}, {mode_D_max}")
            D_edges = np.logspace(np.log10(mode_D_min), np.log10(mode_D_max), num=int(N_bins) + 1)
        
        # geometric bin centers
        D_mids = np.sqrt(D_edges[:-1] * D_edges[1:])
        bin_width = np.log10(D_mids[1]) - np.log10(D_mids[0])
        
        # Map this mode's fractions to the full population species list
        # For each species in pop_species_names, use the fraction from this mode, or 0 if not present
        mode_spec_name_to_frac = dict(
            zip(mode_spec_names, mode_spec_fracs, strict=True)
        )
        pop_aligned_fracs = [mode_spec_name_to_frac.get(n, 0.0) for n in pop_species_names]
        pdf_wrt_logD = norm(loc=np.log10(GMD), scale=np.log10(GSD))
        N_per_bins = pdf_wrt_logD.pdf(np.log10(D_mids)) * bin_width
        if Ntot == 0:
            # Retain zero-number bins without dividing by unrepresented support.
            N_per_bins = np.zeros_like(N_per_bins)
        else:
            support = np.sum(N_per_bins)
            if (not np.all(np.isfinite(N_per_bins))
                    or not np.isfinite(support) or support <= 0):
                raise ValueError(
                    f"Mode {mode_idx} has no finite positive sampled support on "
                    "the diameter grid; adjust the grid or bin resolution."
                )
            # Normalize before scaling to avoid overflow for large finite N.
            N_per_bins = Ntot * (N_per_bins / support)
        for dd, (D, N_per_bin) in enumerate(zip(D_mids, N_per_bins)):
            # Optional debug printing is available via env var if needed
            particle = make_particle(
                D,
                pop_species_list,
                pop_aligned_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet, specdata_path=specdata_path
                )
            part_id += 1
            lognormals_population.set_particle(
                particle, part_id, N_per_bin)
            
    return lognormals_population
    
