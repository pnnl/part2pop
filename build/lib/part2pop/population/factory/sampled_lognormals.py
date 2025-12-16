
"""
Build a sampled lognormal population from one or more lognormal modes.

If N_parts is:
  - a scalar: it is interpreted as the *total* number of samples to draw
    from the multimodal distribution, split across modes in proportion to N.
  - a list: it is interpreted as the number of samples per mode (backward
    compatible behaviour).
"""

from ..base import ParticlePopulation
from part2pop import make_particle
from part2pop.species.registry import get_species
from .registry import register
import numpy as np


@register("sampled_lognormals")
def build(config):

    N_list = np.atleast_1d(config["N"]).astype(float)
    GMD_list = np.atleast_1d(config["GMD"]).astype(float)
    GSD_list = np.atleast_1d(config["GSD"]).astype(float)

    # --- N_parts handling ---------------------------------------------------
    # If N_parts is a list, treat as one entry per mode (backward-compatible).
    # If N_parts is a scalar, treat as total samples from the *mixture*
    # and allocate per mode in proportion to N_list.
    N_parts_cfg = config.get("N_parts", 100)

    if isinstance(N_parts_cfg, list) or isinstance(N_parts_cfg, tuple):
        N_parts_list = np.array(N_parts_cfg, dtype=int)
    else:
        # scalar total samples
        N_parts_total = int(N_parts_cfg)
        if N_parts_total <= 0:
            raise ValueError("N_parts must be positive.")

        # probabilities per mode, proportional to mode number concentration
        p = N_list / N_list.sum()
        N_parts_list = np.round(p * N_parts_total).astype(int)

        # Make sure we don't drop any modes and that the sum matches exactly
        # (adjust by distributing the rounding error).
        N_parts_list = np.maximum(N_parts_list, 1)
        diff = N_parts_total - int(N_parts_list.sum())
        # distribute diff one by one
        i = 0
        while diff != 0:
            if diff > 0:
                N_parts_list[i % len(N_parts_list)] += 1
                diff -= 1
            else:  # diff < 0
                if N_parts_list[i % len(N_parts_list)] > 1:
                    N_parts_list[i % len(N_parts_list)] -= 1
                    diff += 1
            i += 1

    aero_spec_names_list = config["aero_spec_names"]
    aero_spec_fracs_list = config["aero_spec_fracs"]

    # --- sanity check on lengths -------------------------------------------
    lengths = {
        "N": len(N_list),
        "GMD": len(GMD_list),
        "GSD": len(GSD_list),
        "aero_spec_names": len(aero_spec_names_list),
        "aero_spec_fracs": len(aero_spec_fracs_list),
        "N_parts": len(N_parts_list),
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Inconsistent mode counts in sampled_lognormals config: {lengths}. "
            "All of these must have the same length (one entry per mode)."
        )

    species_modifications = config.get("species_modifications", {})
    D_is_wet = config.get("D_is_wet", False)

    # --- Build master species list for the population ----------------------
    pop_species_names = []
    for mode_names in aero_spec_names_list:
        for name in mode_names:
            if name not in pop_species_names:
                pop_species_names.append(name)

    pop_species_list = tuple(
        get_species(spec_name, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    # Initialize an empty population with the correct species list
    sampled_lognormals_population = ParticlePopulation(
        species=pop_species_list,
        spec_masses=[],
        num_concs=[],
        ids=[],
        species_modifications=species_modifications,
    )

    # --- Draw samples per mode and populate --------------------------------
    part_id = 0
    for mode_idx, (Ntot, GMD, GSD, mode_spec_names, mode_spec_fracs, N_parts_mode) in enumerate(
        zip(
            N_list,
            GMD_list,
            GSD_list,
            aero_spec_names_list,
            aero_spec_fracs_list,
            N_parts_list,
        )
    ):
        # map species -> fraction for this mode, then align to population species order
        mode_spec_name_to_frac = dict(zip(mode_spec_names, mode_spec_fracs))
        pop_aligned_fracs = [mode_spec_name_to_frac.get(n, 0.0) for n in pop_species_names]

        # sample diameters for this mode
        Ds = np.exp(
            np.random.normal(
                loc=np.log(GMD),
                scale=np.log(GSD),
                size=int(N_parts_mode),
            )
        )
        # each sample carries equal share of this mode's N
        Ns = np.full(int(N_parts_mode), Ntot / float(N_parts_mode))

        for D, N_per_part in zip(Ds, Ns):
            particle = make_particle(
                D,
                pop_species_list,
                pop_aligned_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet,
            )
            part_id += 1
            sampled_lognormals_population.set_particle(
                particle, part_id, N_per_part
            )

    return sampled_lognormals_population


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Build a binned lognormal population
# @author: Laura Fierce
# """

# from ..base import ParticlePopulation
# from ..utils import expand_compounds_for_population
# from part2pop import make_particle
# from part2pop.species.registry import get_species
# from .registry import register
# from scipy.stats import norm
# import numpy as np
# import matplotlib.pyplot as plt

# @register("sampled_lognormals")
# def build(config):

#     N_list = config['N']
#     GMD_list = config['GMD']
#     GSD_list = config['GSD']
    
#     N_parts = config['N_parts']
#     if type(config['N_parts']) is list:
#         N_parts_list = config.get('N_parts')
#     else:
#         N_parts_val = config.get('N_parts',100)
#         N_parts_list = [N_parts_val]*len(GMD_list)
    
#     aero_spec_names_list = config['aero_spec_names']
#     aero_spec_fracs_list = config['aero_spec_fracs']
    
#     # test to make sure lengths of the different components are correct
#     lengths = {
#         "N": len(N_list),
#         "GMD": len(GMD_list),
#         "GSD": len(GSD_list),
#         "aero_spec_names": len(aero_spec_names_list),
#         "aero_spec_fracs": len(aero_spec_fracs_list),
#         "N_parts": len(N_parts_list),
#     }
#     unique_lengths = set(lengths.values())
#     if len(unique_lengths) != 1:
#         raise ValueError(
#             f"Inconsistent mode counts in sampled_lognormals config: {lengths}. "
#             "All of these must have the same length (one entry per mode)."
#         )
    
#     species_modifications = config.get('species_modifications', {})
#     surface_tension = config.get('surface_tension', 0.072)
#     D_is_wet = config.get('D_is_wet', False)
#     specdata_path = config.get('specdata_path', None)
    
#     # Build master species list for the *population*, preserving order
#     pop_species_names = []
#     for mode_names in aero_spec_names_list:
#         for name in mode_names:
#             if name not in pop_species_names:
#                 pop_species_names.append(name)
#     # Build species objects
#     pop_species_list = tuple(
#         get_species(spec_name, **species_modifications.get(spec_name, {}))
#         for spec_name in pop_species_names
#     )

#     # Create the population object with the right species list
#     sampled_lognormals_population = ParticlePopulation(
#         species=pop_species_list, spec_masses=[], num_concs=[], ids=[],
#         species_modifications=species_modifications
#     )

    
#     part_id = 0
#     for mode_idx, (Ntot, GMD, GSD, mode_spec_names, mode_spec_fracs, N_parts) in enumerate(
#             zip(N_list, GMD_list, GSD_list, aero_spec_names_list, aero_spec_fracs_list, N_parts_list)):
        
#         mode_spec_name_to_frac = dict(zip(mode_spec_names, mode_spec_fracs))
#         pop_aligned_fracs = [mode_spec_name_to_frac.get(n, 0.0) for n in pop_species_names]
        
#         Ds = np.exp(np.random.normal(loc=np.log(GMD), scale=np.log(GSD), size=N_parts))
#         Ns = np.full(N_parts, Ntot / N_parts)
#         for dd, (D, N_per_part) in enumerate(zip(Ds, Ns)):
#             # Optional debug printing is available via env var if needed
#             particle = make_particle(
#                 D,
#                 pop_species_list,
#                 pop_aligned_fracs.copy(),
#                 species_modifications=species_modifications,
#                 D_is_wet=D_is_wet)
#             part_id += 1
#             sampled_lognormals_population.set_particle(
#                 particle, part_id, N_per_part)
#     return sampled_lognormals_population
