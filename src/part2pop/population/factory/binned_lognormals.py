#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a binned lognormal population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from ..utils import expand_compounds_for_population
from part2pop import make_particle
from part2pop.species.registry import get_species
from .registry import register
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

@register("binned_lognormals")
def build(config):
    # fixme: make this +/- certain number of sigmas (rather than min/max diams)
    # D_min = float(config['D_min'])
    # D_max = float(config['D_max'])
        
    N_list = config['N']
    GMD_list = config['GMD']
    GSD_list = config['GSD']
    
    N_bins_list = config['N_bins']
    if type(config['N_bins']) is list:
        N_bins_list = config.get('N_bins')
    else:
        N_bins_val = config.get('N_bins',100)
        N_bins_list = [N_bins_val]*len(GMD_list)
    
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
        if not (global_D_min > 0 and global_D_max > 0 and global_D_min < global_D_max):
            raise ValueError("D_min and D_max must be positive and D_min < D_max.")
    
    aero_spec_names_list = config['aero_spec_names']
    aero_spec_fracs_list = config['aero_spec_fracs']
    # Support compound-like species names (e.g., NaCl, (NH4)2SO4)
    # aero_spec_names_list, aero_spec_fracs_list = expand_compounds_for_population(
    #     aero_spec_names_list, aero_spec_fracs_list
    # )
    species_modifications = config.get('species_modifications', {})
    surface_tension = config.get('surface_tension', 0.072)
    D_is_wet = config.get('D_is_wet', False)
    # specdata_path = config.get('specdata_path', None)
    
    # Build master species list for the *population*, preserving order
    pop_species_names = []
    for mode_names in aero_spec_names_list:
        for name in mode_names:
            if name not in pop_species_names:
                pop_species_names.append(name)
    # Build species objects
    pop_species_list = tuple(
        get_species(spec_name, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    # Create the population object with the right species list
    lognormals_population = ParticlePopulation(
        species=pop_species_list, spec_masses=[], num_concs=[], ids=[],
        species_modifications=species_modifications
    )

    
    part_id = 0
    for mode_idx, (Ntot, GMD, GSD, mode_spec_names, mode_spec_fracs, N_bins) in enumerate(
            zip(N_list, GMD_list, GSD_list, aero_spec_names_list, aero_spec_fracs_list, N_bins_list)):
        # determine bin edges: either global or per-mode
        if global_D_min is not None:
            # use global edges (N_bins bins => N_bins+1 edges)
            D_edges = np.logspace(np.log10(global_D_min), np.log10(global_D_max), num=int(N_bins) + 1)
        else:
            # per-mode edges centered on GMD spanning N_sigmas in log-space
            mode_D_min = np.exp(np.log(GMD) - 0.5 * N_sigmas * np.log(GSD))
            mode_D_max = np.exp(np.log(GMD) + 0.5 * N_sigmas * np.log(GSD))
            if not (mode_D_min > 0 and mode_D_max > 0 and mode_D_min < mode_D_max):
                raise ValueError(f"Invalid mode edges for mode {mode_idx}: {mode_D_min}, {mode_D_max}")
            D_edges = np.logspace(np.log10(mode_D_min), np.log10(mode_D_max), num=int(N_bins) + 1)
        
        # geometric bin centers
        D_mids = np.sqrt(D_edges[:-1] * D_edges[1:])
        bin_width = np.log10(D_mids[1]) - np.log10(D_mids[0])
        
        # Map this mode's fractions to the full population species list
        # For each species in pop_species_names, use the fraction from this mode, or 0 if not present
        mode_spec_name_to_frac = dict(zip(mode_spec_names, mode_spec_fracs))
        pop_aligned_fracs = [mode_spec_name_to_frac.get(n, 0.0) for n in pop_species_names]
        pdf_wrt_logD = norm(loc=np.log10(GMD), scale=np.log10(GSD))
        N_per_bins = pdf_wrt_logD.pdf(np.log10(D_mids)) * bin_width
        N_per_bins = float(Ntot) * N_per_bins / np.sum(N_per_bins)
        for dd, (D, N_per_bin) in enumerate(zip(D_mids, N_per_bins)):
            # Optional debug printing is available via env var if needed
            particle = make_particle(
                D,
                pop_species_list,
                pop_aligned_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet#, specdata_path=specdata_path
                )
            part_id += 1
            lognormals_population.set_particle(
                particle, part_id, N_per_bin)
            
    return lognormals_population
    
