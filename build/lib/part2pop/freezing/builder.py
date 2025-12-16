#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 14:35:25 2025

@author: beel083
"""

from .factory.registry import discover_morphology_types
from .base import FreezingPopulation
import numpy as np


class FreezingParticleBuilder:
    """Construct an optical particle instance from a config.

    Parameters
    ----------
    config : dict
        Configuration dictionary with a 'type' key indicating morphology.
    """
    def __init__(self, config):
        self.config = config
    
    def build(self, base_particle):
        type_name = self.config.get("morphology")
        if not type_name:
            raise ValueError("Config must include a 'morphology' key.")
        types = discover_morphology_types()
        if type_name not in types:
            raise ValueError(f"Unknown freezing morphology type: {type_name}")
        cls_or_factory = types[type_name]
        # Expect a class or callable that accepts (base_particle, config)
        return cls_or_factory(base_particle, self.config)


def build_freezing_particle(base_particle, config):
    """Helper: build and return an optical particle from base particle and config."""
    return FreezingParticleBuilder(config).build(base_particle)

def build_freezing_population(base_population, config, T=None):
    """Build a FreezingPopulation from a base ParticlePopulation and config.

    Parameters
    ----------
    base_population : ParticlePopulation
        Base population containing species, masses, concentrations, and ids.
    config : dict
        Optics configuration (rh_grid, wvl_grid, and morphology type).

    Returns
    -------
    FreezingPopulation
        INP properties of the population.
    """
    
    # Pass the base population so FreezingPopulation can inherit ids/num_concs/etc.
    
    T_units = config.get("T_units", None)
    if not T:
        T = config.get("T_grid", None)
        T = np.array(T)
    
    if T_units=="C":
        freezing_population = FreezingPopulation(base_population, T+273.15)
        for part_id in base_population.ids:
            base_particle = base_population.get_particle(part_id)
            freezing_particle = build_freezing_particle(base_particle, config)
            freezing_population.add_freezing_particle(freezing_particle, part_id, T+273.15)
    elif T_units=="K":
        freezing_population = FreezingPopulation(base_population, T)
        for part_id in base_population.ids:
            base_particle = base_population.get_particle(part_id)
            freezing_particle = build_freezing_particle(base_particle, config)
            freezing_population.add_freezing_particle(freezing_particle, part_id, T)
    else:
        raise ValueError(f"Unknown or unspecified temperature unit: {T_units}")
    
    return freezing_population


