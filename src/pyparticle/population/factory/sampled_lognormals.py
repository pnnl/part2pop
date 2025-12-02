#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a binned lognormal population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from ..utils import expand_compounds_for_population
from pyparticle import make_particle
from pyparticle.species.registry import get_species
from .registry import register
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

@register("sampled_lognormals")
def build(config):
    # fixme: make this +/- certain number of sigmas (rather than min/max diams)
    # D_min = float(config['D_min'])
    # D_max = float(config['D_max'])
        
    N_list = config['N']
    GMD_list = config['GMD']
    GSD_list = config['GSD']
    
    N_parts = config['N_parts']
    if type(config['N_parts']) is list:
        N_parts_list = config.get('N_parts')
    else:
        N_parts_val = config.get('N_parts',100)
        N_parts_list = [N_parts_val]*len(GMD_list)
    
    aero_spec_names_list = config['aero_spec_names']
    aero_spec_fracs_list = config['aero_spec_fracs']
    
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
        get_species(spec_name, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    # Create the population object with the right species list
    sampled_lognormals_population = ParticlePopulation(
        species=pop_species_list, spec_masses=[], num_concs=[], ids=[],
        species_modifications=species_modifications
    )

    
    part_id = 0
    for mode_idx, (Ntot, GMD, GSD, mode_spec_names, mode_spec_fracs, N_parts) in enumerate(
            zip(N_list, GMD_list, GSD_list, aero_spec_names_list, aero_spec_fracs_list, N_parts_list)):
        # # determine bin edges: either global or per-mode
        # if global_D_min is not None:
        #     # use global edges (N_bins bins => N_bins+1 edges)
        #     D_edges = np.logspace(np.log10(global_D_min), np.log10(global_D_max), num=int(N_bins) + 1)
        # else:
        #     # per-mode edges centered on GMD spanning N_sigmas in log-space
        #     mode_D_min = np.exp(np.log(GMD) - 0.5 * N_sigmas * np.log(GSD))
        #     mode_D_max = np.exp(np.log(GMD) + 0.5 * N_sigmas * np.log(GSD))
        #     if not (mode_D_min > 0 and mode_D_max > 0 and mode_D_min < mode_D_max):
        #         raise ValueError(f"Invalid mode edges for mode {mode_idx}: {mode_D_min}, {mode_D_max}")
        #     D_edges = np.logspace(np.log10(mode_D_min), np.log10(mode_D_max), num=int(N_bins) + 1)
        
        # # geometric bin centers
        # D_mids = np.sqrt(D_edges[:-1] * D_edges[1:])
        # bin_width = np.log10(D_mids[1]) - np.log10(D_mids[0])
        
        # Map this mode's fractions to the full population species list
        # For each species in pop_species_names, use the fraction from this mode, or 0 if not present
        
        mode_spec_name_to_frac = dict(zip(mode_spec_names, mode_spec_fracs))
        pop_aligned_fracs = [mode_spec_name_to_frac.get(n, 0.0) for n in pop_species_names]
        # pdf_wrt_logD = norm(loc=np.log10(GMD), scale=np.log10(GSD))
        
        # N_per_bins = pdf_wrt_logD.pdf(np.log10(D_mids)) * bin_width
        # N_per_bins = float(Ntot) * N_per_bins / np.sum(N_per_bins)
        Ds = np.exp(np.random.normal(loc=np.log(GMD), scale=np.log(GSD), size=N_parts))
        Ns = np.full(N_parts, Ntot / N_parts)
        for dd, (D, N_per_part) in enumerate(zip(Ds, Ns)):
            # Optional debug printing is available via env var if needed
            particle = make_particle(
                D,
                pop_species_list,
                pop_aligned_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet, specdata_path=specdata_path)
            part_id += 1
            sampled_lognormals_population.set_particle(
                particle, part_id, N_per_part)
    return sampled_lognormals_population
