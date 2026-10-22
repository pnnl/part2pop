#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a monodisperse population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from ..utils import normalize_population_config
from part2pop import make_particle
from part2pop.species.registry import get_species
from part2pop.species.resolution import resolve_species_name_rows
import numpy as np
from .registry import register

@register("monodisperse")
def build(config):
    config = normalize_population_config(config)
    aero_spec_names = resolve_species_name_rows(config['aero_spec_names'])
    species_modifications = config.get('species_modifications', {})
    N = config['N']
    D = config['D']
    aero_spec_fracs = config['aero_spec_fracs']
    D_is_wet = config.get('D_is_wet', False)
    specdata_path = config.get('specdata_path', None)
    classes = config.get('classes', None)
    
    # Build master species list for the *population*, preserving order
    pop_species_names = []
    for part_names in aero_spec_names:
        for name in part_names:
            if name not in pop_species_names:
                pop_species_names.append(name)

    species_list = tuple(
        get_species(spec_name, specdata_path, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    if len(D) != len(N) or len(aero_spec_fracs) != len(N):
        raise ValueError("D, N, and aero_spec_fracs must have one entry per particle")
    if len(aero_spec_names) == 1 and len(N) > 1:
        # Preserve the historical shorthand of one shared species-name row.
        name_rows = [aero_spec_names[0] for _ in N]
    elif len(aero_spec_names) == len(N):
        name_rows = aero_spec_names
    else:
        raise ValueError("aero_spec_names must have one row or one row per particle")

    monodisp_population = ParticlePopulation(
        species=species_list, spec_masses=[], num_concs=[], ids=[],
        classes=classes, species_modifications=species_modifications)
    for i, (names, fractions) in enumerate(zip(name_rows, aero_spec_fracs)):
        if len(names) != len(fractions) or len(set(names)) != len(names):
            raise ValueError(f"Particle {i} has mismatched or duplicate species names/fractions")
        # Fractions are positional within each particle's resolved species row,
        # not within the population-wide species ordering. Map by canonical name;
        # absent species contribute zero mass fraction for that particle.
        fraction_by_name = dict(zip(names, fractions))
        aligned_fractions = [fraction_by_name.get(name, 0.0) for name in pop_species_names]
        particle = make_particle(
            D[i], species_list, aligned_fractions,
            species_modifications=species_modifications,
            specdata_path=specdata_path, D_is_wet=D_is_wet)
        monodisp_population.set_particle(particle, i, N[i])
    return monodisp_population

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Build a monodisperse population
# @author: Laura Fierce
# """

# from ..utils import normalize_population_config
# from .helpers.assembly import assemble_population_from_mass_fractions
# from .registry import register

# @register("monodisperse")
# def build(config):
#     config = normalize_population_config(config)
#     diameters = config['D']
#     number_concentrations = config['N']
#     aero_spec_names = config['aero_spec_names']
#     aero_spec_fracs = config['aero_spec_fracs']

#     n_particles = len(number_concentrations)

#     # Backward compatibility:
#     # allow one shared species-name row with multiple explicit particle entries.
#     if len(aero_spec_names) == 1 and n_particles > 1:
#         aero_spec_names = [list(aero_spec_names[0]) for _ in range(n_particles)]

#     return assemble_population_from_mass_fractions(
#         diameters=diameters,
#         number_concentrations=number_concentrations,
#         species_names=aero_spec_names,
#         mass_fractions=aero_spec_fracs,
#         ids=None,
#         classes=config.get('classes', None),
#         species_modifications=config.get('species_modifications', {}),
#         D_is_wet=config.get('D_is_wet', False),
#         specdata_path=config.get('specdata_path', None),
#     )
