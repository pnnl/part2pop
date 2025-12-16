#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a monodisperse population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from part2pop import make_particle
from part2pop.species.registry import get_species
import numpy as np
from .registry import register

@register("monodisperse")
def build(config):
    aero_spec_names = config['aero_spec_names']
    species_modifications = config.get('species_modifications', {})
    N = config['N']
    D = config['D']
    aero_spec_fracs = config['aero_spec_fracs']
    D_is_wet = config.get('D_is_wet', False)
    specdata_path = config.get('specdata_path', None)
    
    # Build master species list for the *population*, preserving order
    pop_species_names = []
    for part_names in aero_spec_names:
        for name in part_names:
            if name not in pop_species_names:
                pop_species_names.append(name)

    species_list = tuple(
        get_species(spec_name, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    monodisp_population = ParticlePopulation(
        species=species_list, spec_masses=[], num_concs=[], ids=[],
        species_modifications=species_modifications)
    for i in range(len(N)):
        particle = make_particle(
            D[i],
            species_list,
            aero_spec_fracs[i].copy(),
            species_modifications=species_modifications,
            D_is_wet=D_is_wet)
        part_id = i
        monodisp_population.set_particle(
            particle, part_id, N[i])
    return monodisp_population
