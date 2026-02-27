#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from EDX measurements. Need to supply a .csv file with the 
following columns:
@author: Payton Beeler
"""

from .registry import register
from typing import Any, Dict
from part2pop.population.base import ParticlePopulation
import pandas as pd
import numpy as np
import warnings

class ElementMasses:
    """Represent molar mass of elements that might be in EDX analysis.

    Attributes
    ----------
    species : tuple[AerosolSpecies,...]
        Sequence of species objects that make up the particle.
    molec_mass : tuple[float,...]
        Mass of each species in SI units (kg).
    """
    def __init__(self):
        self.O = 16.0e-3
        self.Mn = 54.94e-3
        self.Fe = 55.85e-3
        self.Mg = 24.305e-3
        self.Al = 27.0e-3
        self.Si = 28.085e-3
        self.K = 39.09e-3
        self.Ca = 40.078e-3
        
class Population_MassFracs:
    def __init__(self, diameters, elements, mass_fractions, ptypes):
        self.D = diameters
        self.elements = elements
        self.mass_fractions = mass_fractions
        self.ptype = ptypes


@register("edx_observations")
def build(config: Dict[str, Any]) -> ParticlePopulation:

    required = ["edx_file", "elements"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"edx_observations missing required config keys: {missing}")

    raw_population = read_edx_file(config)

    aerospecs = config.get('aerosol_species', None)
    for ptype, MassFracs in zip(raw_population.ptype, raw_population.mass_fractions):
        if ptype == 'dust':
            masses = sample_dust_particle(aerospecs, MassFracs, raw_population.elements)
            print(masses)
            import sys
            sys.exit()


    # config = normalize_population_config(config)
    # required = ["aimms_file", "splat_file", "ams_file", "z", "dz", "splat_species", "mass_thresholds"]
    # missing = [k for k in required if k not in config]
    # if missing:
    #     raise KeyError(f"hiscale_observations missing required config keys: {missing}")
    # if "fims_file" not in config and "beasd_file" not in config:
    #     raise KeyError("hiscale_observations requires either 'fims_file' or 'beasd_file' in config.")
    # if "fims_file" in config and "fims_bins_file" not in config:
    #     raise KeyError("hiscale_observations with 'fims_file' requires 'fims_bins_file' in config.")
    
    return 1


def read_edx_file(config: Dict[str, Any]) -> Dict[str, float]:
    filename = config["edx_file"]
    extension = filename.split('.')[-1]
    if extension != 'csv':
        raise ValueError(f"edx_file must be .csv format supplied format is .{extension}")
    data = pd.read_csv(filename)
    particle_massfracs = np.zeros((len(data), len(config['elements'])))
    elements = np.array(config['elements'])

    # find element columns
    for jj, (spec) in enumerate(elements):
        idx = None
        for ii, (column) in enumerate(data.keys()):
            if column.split("_")[-1] == spec:
                idx = ii
                particle_massfracs[:,jj]=np.array(data[column])
        if not idx:
            raise KeyError(f"Could not identify column for {spec} in {filename}")

    # find size column
    matches = [k for k in data if "diam" in k.lower()]
    if len(matches)==0:
        raise KeyError(f"Could not identify diameter column in {filename}")
    elif len(matches)>1:
        warnings.warn(f"Possible diameter columns identified: {matches}. Proceeding using {matches[0]}.", UserWarning)
    particle_diameters = 1e-6*np.array(data[matches[0]]) # assumes that diameters are always reported in micron

    # find class column
    matches = [k for k in data if any(term in k.lower() for term in ("label", "class", "type"))]
    if len(matches)==0:
        raise KeyError(f"Could not identify classification column in {filename}")
    elif len(matches)>1:
        warnings.warn(f"Possible classification columns identified: {matches}. Proceeding using {matches[0]}.", UserWarning)
    ptypes = np.array(data[matches[0]], dtype='str')
    ptypes = np.char.lower(ptypes) # make everything lowercase

    return Population_MassFracs(particle_diameters, elements, particle_massfracs, ptypes)


def sample_dust_particle(aerospecs: list[str], mass_fraction: np.ndarray[float], elements: np.ndarray[str]) -> list[float]:
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = dict(zip(elements, mass_fraction))
    molec_masses = ElementMasses()

    # dust = Mg + Al + Si + P + K + Ca + Mn + Fe + Zn 
    # assume these elements are present as MgO, Al2O3, SiO2, K2O, CaO, Fe2O3, MnO
    dust_O_fraction = (
        data_dict['Mg']*(molec_masses.O/molec_masses.Mg)
        + data_dict['Al']*((3*molec_masses.O)/(2*molec_masses.Al))
        + data_dict['Si']*((2*molec_masses.O)/molec_masses.Si)
        + data_dict['K']*(molec_masses.O/(2*molec_masses.K))
        + data_dict['Ca']*(molec_masses.O/molec_masses.Ca)
        + data_dict['Fe']*((3*molec_masses.O)/(2*molec_masses.Fe))
        + data_dict['Mn']*(molec_masses.O/molec_masses.Mn)
    )

    print(dust_O_fraction, data_dict['O'])
    # dust_fractions = (
    #     raw_data['Mg'][idx]
    #     + raw_data['Al'][idx]
    #     + raw_data['Si'][idx]
    #     + raw_data['K'][idx]
    #     + raw_data['Ca'][idx]
    #     + raw_data['Fe'][idx]
    #     + raw_data['Mn'][idx]
    #     + dust_O_fraction
    # )
    

    return sampled_masses

