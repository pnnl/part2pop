
"""
Build a sampled lognormal population from one or more lognormal modes.

If N_parts is:
  - a scalar: it is interpreted as the *total* number of samples to draw
    from the multimodal distribution, split across modes in proportion to N.
  - a list: it is interpreted as the number of samples per mode (backward
    compatible behaviour).
"""

from ..base import ParticlePopulation
from ..utils import normalize_population_config
from .registry import register
from part2pop import make_particle
from part2pop.species.registry import get_species

import numpy as np


def _sample_count(value):
    """Validate raw counts before the generic normalizer can truncate them."""
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError("N_parts must contain finite positive integer sample counts.")
    try:
        numeric = value if isinstance(value, (int, np.integer)) else float(value)
        count = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("N_parts must contain finite positive integer sample counts.") from exc
    if (count <= 0 or count > np.iinfo(np.intp).max
            or not np.isfinite(numeric) or numeric != count):
        raise ValueError("N_parts must contain finite positive integer sample counts.")
    return count


@register("sampled_lognormals")
def build(config):
    raw_n_parts = config.get("N_parts", 100)
    config = dict(config)
    config.pop("N_parts", None)
    config = normalize_population_config(config)
    seed = config.get("seed", None)
    rng = np.random.default_rng(seed)
    N_list = np.atleast_1d(config["N"]).astype(float)
    GMD_list = np.atleast_1d(config["GMD"]).astype(float)
    GSD_list = np.atleast_1d(config["GSD"]).astype(float)

    # --- N_parts handling ---------------------------------------------------
    # If N_parts is a list, treat as one entry per mode (backward-compatible).
    # If N_parts is a scalar, treat as total samples from the *mixture*
    # and allocate per mode in proportion to N_list.
    if (isinstance(raw_n_parts, (str, bytes)) or np.isscalar(raw_n_parts)
            or isinstance(raw_n_parts, np.ndarray) and raw_n_parts.ndim == 0):
        N_parts_cfg = _sample_count(raw_n_parts)
    else:
        try:
            N_parts_cfg = [_sample_count(value) for value in raw_n_parts]
        except TypeError as exc:
            raise ValueError("N_parts must be an integer or an iterable of integers.") from exc
    specdata_path = config.get("specdata_path", None)

    # Finite entries can still overflow their sum. Reject that configuration
    # rather than silently losing samples when normalized quotas become zero.
    with np.errstate(over="ignore", invalid="ignore"):
        total_concentration = float(np.sum(N_list))
    if (not np.all(np.isfinite(N_list)) or np.any(N_list < 0)
            or not np.isfinite(total_concentration) or total_concentration <= 0):
        raise ValueError("Mode number concentrations must be finite, nonnegative, and have finite positive total.")

    if isinstance(N_parts_cfg, (list, tuple)):
        N_parts_list = np.asarray(N_parts_cfg, dtype=int)
        if np.any(N_parts_list <= 0):
            raise ValueError("Each mode must have a positive N_parts count.")
    else:
        N_parts_total = int(N_parts_cfg)
        if N_parts_total <= 0:
            raise ValueError("N_parts must be positive.")
        n_modes = len(N_list)
        if N_parts_total < n_modes:
            raise ValueError("N_parts must be at least the number of modes so each mode has a sample.")

        # Reserve one sample per mode, then apportion the remaining samples
        # by largest remainder. This has a strict O(n_modes log n_modes) bound.
        remaining = N_parts_total - n_modes
        quotas = (N_list / total_concentration) * remaining
        extra = np.floor(quotas).astype(int)
        remainder = remaining - int(extra.sum())
        if remainder:
            priority = np.argsort(-(quotas - extra), kind="stable")
            extra[priority[:remainder]] += 1
        N_parts_list = extra + 1

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
        get_species(spec_name, specdata_path, **species_modifications.get(spec_name, {}))
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
            rng.normal(
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
