#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from EDX measurements. Need to supply a .csv file with the 
following columns:
@author: Payton Beeler
"""

from .registry import register
from .helpers.assembly import assemble_population_from_mass_fractions
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
        self.S = 32.065e-3
        self.Na = 22.99e-3
        self.Cl = 35.45e-3
        
class Population_MassFracs:
    def __init__(self, diameters, elements, mass_fractions, ptypes):
        self.D = diameters
        self.elements = elements
        self.mass_fractions = mass_fractions
        self.ptype = ptypes


@register("edx_observations")
def build(config: Dict[str, Any]) -> ParticlePopulation:

    required = ["edx_file"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"edx_observations missing required config keys: {missing}")
    elements = config.get("elements", ['C','N','O','Na','Mg','Al','Si','P','S','Cl','K','Ca','Mn','Fe','Zn'])

    raw_population = read_edx_file(config, elements)
    particle_classes = []
    aero_spec_names = config.get("aerosol_species", ['SO4','OIN','OC','Na','Cl','biological'])
    aero_spec_masses = np.zeros((len(raw_population.ptype), len(aero_spec_names)))
    for ii, (ptype, MassFracs) in enumerate(zip(raw_population.ptype, raw_population.mass_fractions)):
        particle_classes.append(ptype)
        if ptype == 'biological':
            aero_spec_masses[ii] = sample_bio_particle(aero_spec_names, MassFracs, raw_population.elements)
        elif ptype in ('carbonaceous', 'carbonaceous mixed dust','dust-carbonaceous'):
            aero_spec_masses[ii] = sample_carbonaceous_particle(aero_spec_names, MassFracs, raw_population.elements)
        else:
            aero_spec_masses[ii] = sample_particle(aero_spec_names, MassFracs, raw_population.elements)

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


def read_edx_file(config: Dict[str, Any], elements: list[str]) -> Dict[str, float]:
    filename = config["edx_file"]
    extension = filename.split('.')[-1]
    if extension != 'csv':
        raise ValueError(f"edx_file must be .csv format supplied format is .{extension}")
    data = pd.read_csv(filename)
    particle_massfracs = np.zeros((len(data), len(elements)))
    elements = np.array(elements)

    # find element columns
    for jj, (spec) in enumerate(elements):
        idx = None
        for ii, (column) in enumerate(data.keys()):
            if column.split("_")[-1] == spec:
                idx = ii
                particle_massfracs[:,jj]=np.array(data[column])
            elif column == spec:
                idx = ii
                particle_massfracs[:,jj]=np.array(data[column])
        if not idx:
            raise KeyError(f"Could not identify column for {spec} in {filename}")

    # calculate mass fraction if in percent
    if np.isclose(np.average(np.sum(particle_massfracs, axis=1)), 100.0):
        particle_massfracs/=100.0

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


def sample_particle(aerospecs: list[str], mass_fraction: np.ndarray[float], elements: np.ndarray[str]) -> list[float]:
    aerospecs=np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = dict(zip(elements, mass_fraction))
    molec_masses = ElementMasses()

    # Assume that all sulfur exists as SO4
    sulfate_O_fraction = data_dict['S']*((4*molec_masses.O)/molec_masses.S)
    if sulfate_O_fraction <= data_dict['O']:
        sulfate_mass_fraction = data_dict['S']+sulfate_O_fraction
    else:
        sulfate_mass_fraction = data_dict['S']+data_dict['O']
        sulfate_O_fraction = data_dict['O']
    try:
        idx = np.where(aerospecs == 'SO4')[0][0]
        sampled_masses[idx]=sulfate_mass_fraction
    except:
        raise ValueError(f"Could not find SO4 in provided aerospecs: {aerospecs}")

    # dust = Mg + Al + Si + K + Ca + Mn + Fe + Zn
    # assume these elements are present as MgO, Al2O3, SiO2, K2O, CaO, Fe2O3, MnO
    # unless the particle does not contain that much oxygen. In that case, assume the
    # rest of the particle's oxygen forms metal oxides.
    dust_O_fraction = (
        data_dict['Mg']*(molec_masses.O/molec_masses.Mg)
        + data_dict['Al']*((3*molec_masses.O)/(2*molec_masses.Al))
        + data_dict['Si']*((2*molec_masses.O)/molec_masses.Si)
        + data_dict['K']*(molec_masses.O/(2*molec_masses.K))
        + data_dict['Ca']*(molec_masses.O/molec_masses.Ca)
        + data_dict['Fe']*((3*molec_masses.O)/(2*molec_masses.Fe))
        + data_dict['Mn']*(molec_masses.O/molec_masses.Mn)
    )
    if sulfate_O_fraction+dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']-sulfate_O_fraction
    dust_mass_fraction = dust_O_fraction
    for kk in ['Mg','Al','Si','K','Ca','Fe','Mn','Zn']:
        dust_mass_fraction += data_dict[kk]
    try:
        idx = np.where(aerospecs == 'OIN')[0][0]
        sampled_masses[idx]=dust_mass_fraction
    except:
        raise ValueError(f"Could not find OIN in provided aerospecs: {aerospecs}")

    # Assume that the remaining carbon, oxygen, phosphorous, and nitrogen are organic
    organic_mass_fraction = (
        data_dict['C'] + data_dict['N'] + data_dict['P'] 
        + data_dict['O'] - sulfate_O_fraction - dust_O_fraction
    )
    try:
        idx = np.where(aerospecs == 'OC')[0][0]
        sampled_masses[idx]=organic_mass_fraction
    except:
        raise ValueError(f"Could not find OC in provided aerospecs: {aerospecs}")

    # Assume that all the Na and Cl exist as NaCl
    try:
        idx = np.where(aerospecs == 'Na')[0][0]
        sampled_masses[idx]=data_dict['Na']
        idx = np.where(aerospecs == 'Cl')[0][0]
        sampled_masses[idx]=data_dict['Cl']
    except:
        raise ValueError(f"Could not find Na or Cl in provided aerospecs: {aerospecs}")

    if np.sum(sampled_masses)>0.99 and np.sum(sampled_masses)<1.01:
        sampled_masses/=np.sum(sampled_masses)
    else:
        raise ValueError(f"Sampled mass fractions sum to {np.sum(sampled_masses)}.")
    
    return sampled_masses


def sample_bio_particle(aerospecs: list[str], mass_fraction: np.ndarray[float], elements: np.ndarray[str]) -> np.ndarray[float]:
    
    aerospecs=np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = dict(zip(elements, mass_fraction))
    molec_masses = ElementMasses()

    # dust = Al + Si
    # assume these elements are present as Al2O3 and SiO2
    # unless the particle does not contain that much oxygen. In that case, assume the
    # rest of the particle's oxygen forms metal oxides.
    dust_O_fraction = (
        + data_dict['Al']*((3*molec_masses.O)/(2*molec_masses.Al))
        + data_dict['Si']*((2*molec_masses.O)/molec_masses.Si)
    )
    if dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']
    dust_mass_fraction = dust_O_fraction
    for kk in ['Al','Si']:
        dust_mass_fraction += data_dict[kk]
    try:
        idx = np.where(aerospecs == 'OIN')[0][0]
        sampled_masses[idx]=dust_mass_fraction
    except:
        raise ValueError(f"Could not find OIN in provided aerospecs: {aerospecs}")
    
    # Assume that all the Na and Cl exist as NaCl
    try:
        idx = np.where(aerospecs == 'Na')[0][0]
        sampled_masses[idx]=data_dict['Na']
        idx = np.where(aerospecs == 'Cl')[0][0]
        sampled_masses[idx]=data_dict['Cl']
    except:
        raise ValueError(f"Could not find Na or Cl in provided aerospecs: {aerospecs}")
    
    # usually found in biological particles: C, N, O, P, S, K, Mg, Ca, Fe, Mn, and Zn
    # Not usually found in biological particles: Na+Cl (salt coating), Si/Al (dust coating).
    # Assume that bio mass = C + N + P + S + K + Mg + Ca + Fe + Mn + Zn + remaining O
    bio_mass_fraction =  (
        + data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['S'] + data_dict['K'] + data_dict['Mg']
        + data_dict['Ca'] + data_dict['Fe'] + data_dict['Mn']
        + data_dict['Zn'] + data_dict['O'] - dust_O_fraction
    )
    try:
        idx = np.where(aerospecs == 'biological')[0][0]
        sampled_masses[idx]=bio_mass_fraction
    except:
        raise ValueError(f"Could not find bio in provided aerospecs: {aerospecs}")
   
    # make sure sums to one
    if np.sum(sampled_masses)>0.99 and np.sum(sampled_masses)<1.01:
        sampled_masses/=np.sum(sampled_masses)
    else:
        raise ValueError(f"Sampled mass fractions sum to {np.sum(sampled_masses)}.")

    return sampled_masses
    
    
def sample_carbonaceous_particle(aerospecs: list[str], mass_fraction: np.ndarray[float], elements: np.ndarray[str]) -> list[float]:
    aerospecs=np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = dict(zip(elements, mass_fraction))
    molec_masses = ElementMasses()

    # Assume that all sulfur exists as SO4
#    sulfate_O_fraction = data_dict['S']*((4*molec_masses.O)/molec_masses.S)
#    if sulfate_O_fraction <= data_dict['O']:
#        sulfate_mass_fraction = data_dict['S']+sulfate_O_fraction
#    else:
#        sulfate_mass_fraction = data_dict['S']+data_dict['O']
#        sulfate_O_fraction = data_dict['O']
#    try:
#        idx = np.where(aerospecs == 'SO4')[0][0]
#        sampled_masses[idx]=sulfate_mass_fraction
#    except:
#        raise ValueError(f"Could not find SO4 in provided aerospecs: {aerospecs}")

    # dust = Mg + Al + Si + K + Ca + Mn + Fe + Zn
    # assume these elements are present as MgO, Al2O3, SiO2, K2O, CaO, Fe2O3, MnO
    # unless the particle does not contain that much oxygen. In that case, assume the
    # rest of the particle's oxygen forms metal oxides.
    dust_O_fraction = (
        data_dict['Mg']*(molec_masses.O/molec_masses.Mg)
        + data_dict['Al']*((3*molec_masses.O)/(2*molec_masses.Al))
        + data_dict['Si']*((2*molec_masses.O)/molec_masses.Si)
        + data_dict['K']*(molec_masses.O/(2*molec_masses.K))
        + data_dict['Ca']*(molec_masses.O/molec_masses.Ca)
        + data_dict['Fe']*((3*molec_masses.O)/(2*molec_masses.Fe))
        + data_dict['Mn']*(molec_masses.O/molec_masses.Mn)
    )
    if dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']
    dust_mass_fraction = dust_O_fraction
    for kk in ['Mg','Al','Si','K','Ca','Fe','Mn','Zn']:
        dust_mass_fraction += data_dict[kk]
    try:
        idx = np.where(aerospecs == 'OIN')[0][0]
        sampled_masses[idx]=dust_mass_fraction
    except:
        raise ValueError(f"Could not find OIN in provided aerospecs: {aerospecs}")

    # Assume that the remaining carbon, oxygen, phosphorous, nitrogen, and sulfur are organic
    organic_mass_fraction = (
        data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['S'] + data_dict['O'] - dust_O_fraction
    )
    try:
        idx = np.where(aerospecs == 'OC')[0][0]
        sampled_masses[idx]=organic_mass_fraction
    except:
        raise ValueError(f"Could not find OC in provided aerospecs: {aerospecs}")

    # Assume that all the Na and Cl exist as NaCl
    try:
        idx = np.where(aerospecs == 'Na')[0][0]
        sampled_masses[idx]=data_dict['Na']
        idx = np.where(aerospecs == 'Cl')[0][0]
        sampled_masses[idx]=data_dict['Cl']
    except:
        raise ValueError(f"Could not find Na or Cl in provided aerospecs: {aerospecs}")

    if np.sum(sampled_masses)>0.99 and np.sum(sampled_masses)<1.01:
        sampled_masses/=np.sum(sampled_masses)
    else:
        raise ValueError(f"Sampled mass fractions sum to {np.sum(sampled_masses)}.")
    
    return sampled_masses

