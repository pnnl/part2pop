#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from EDX measurements. Need to supply a .csv file with the 
following columns:
@author: Payton Beeler
"""

from .registry import register
from .helpers.assembly import assemble_population_from_mass_fractions
from .helpers.edx import read_edx_file, reconstruct_edx_species_mass_fractions
from typing import Any, Dict
from part2pop.population.base import ParticlePopulation
import numpy as np


@register("edx_observations")
def build(config: Dict[str, Any]) -> ParticlePopulation:

    required = ["edx_file"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"edx_observations missing required config keys: {missing}")
    elements = config.get("elements", ['C','N','O','Na','Mg','Al','Si','P','S','Cl','K','Ca','Mn','Fe','Zn'])

    raw_population = read_edx_file(config, elements)
    aero_spec_names = config.get("aerosol_species", ['SO4','OIN','OC','Na','Cl','biological'])
    aero_spec_masses, particle_classes = reconstruct_edx_species_mass_fractions(raw_population, aero_spec_names)

    # make the particle population from explicit per-particle rows
    new_aero_spec_names = [aero_spec_names.copy() for _ in range(len(aero_spec_masses))]
    species_modifications = config.get("species_modifications", {})
    D_is_wet = config.get("D_is_wet", False)
    specdata_path = config.get("specdata_path")
    particle_population = assemble_population_from_mass_fractions(
        diameters=raw_population.D,
        number_concentrations=np.ones(len(aero_spec_masses)),
        species_names=new_aero_spec_names,
        mass_fractions=aero_spec_masses,
        classes=particle_classes,
        species_modifications=species_modifications,
        D_is_wet=D_is_wet,
        specdata_path=specdata_path,
    )
    return particle_population

