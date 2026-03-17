#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from a PARTMC NetCDF file
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from part2pop import make_particle_from_masses
from part2pop.species.registry import get_species
import numpy as np
import netCDF4
import os
from pathlib import Path
from .registry import register

@register("partmc") # only registers if netCDF4 is available
def build(config):
    partmc_dir = Path(config['partmc_dir'])
    timestep = config['timestep']
    repeat = config['repeat']
    n_particles = config.get('n_particles', None)
    N_tot = config.get('N_tot', None)
    species_modifications = config.get('species_modifications', {})
    specdata_path = config.get('specdata_path', None)
    suppress_warning = config.get('suppress_warning', True)
    add_mixing_ratios = config.get('add_mixing_ratios', True)

    partmc_filepath = get_ncfile(partmc_dir / 'out', timestep, repeat)
    currnc = netCDF4.Dataset(partmc_filepath)
    
    
    aero_spec_names = currnc.variables['aero_species'].names.split(',')
    # if '.' in aero_spec_names[0]:
    #     aero_spec_names = map_camp_specs(currnc.variables['aero_species'].names.split(','))
    # else:
    #     aero_spec_names = currnc.variables['aero_species'].names.split(',')
    # Get AerosolSpecies objects with modifications if any
    species_list = tuple(
        get_species(name, **species_modifications.get(name, {}))
        for name in aero_spec_names
    )
    spec_masses = np.array(currnc.variables['aero_particle_mass'][:])
    part_ids = np.array([one_id for one_id in currnc.variables['aero_id'][:]], dtype=int)
    
    if 'aero_num_conc' in currnc.variables.keys():
        num_concs = currnc.variables['aero_num_conc'][:]
    else:
        num_concs = 1. / currnc.variables['aero_comp_vol'][:]

    if N_tot is None:
        N_tot = np.sum(num_concs)

    if n_particles is None:
        idx = np.arange(len(part_ids))
    elif n_particles <= len(part_ids):
        idx = np.random.choice(np.arange(len(part_ids)), size=n_particles, replace=False)
    else:
        raise IndexError('n_particles > len(part_ids)')
    
    partmc_population = ParticlePopulation(
        species=species_list,
        spec_masses=[],
        num_concs=[],
        ids=[],
        species_modifications=species_modifications,
    )
    for ii in idx:
        particle = make_particle_from_masses(
            aero_spec_names,
            spec_masses[:, ii],
            species_modifications=species_modifications,
        )
        partmc_population.set_particle(
            particle, part_ids[ii], num_concs[ii] * N_tot / np.sum(num_concs[idx]), suppress_warning=suppress_warning
        )

    if add_mixing_ratios:
        gas_mixing_ratios = np.array(currnc.variables['gas_mixing_ratio'][:])
        partmc_population.gas_mixing_ratios = gas_mixing_ratios
    return partmc_population


def map_camp_specs(camp_spec_names):
    spec_names = []
    for spec_name in camp_spec_names:
        split_specs = spec_name.split('.')
        spec_names.append(split_specs[-1])
        # spec_names.append(spec_name)
    # print(spec_names)
    return spec_names

def get_ncfile(partmc_output_dir, timestep, repeat):

    if not os.path.exists(partmc_output_dir):
        raise FileNotFoundError(f"PartMC output directory {partmc_output_dir} does not exist.")
    for root, dirs, files in os.walk(partmc_output_dir):
        f = files[0]
    if f.startswith('urban_plume_wc_'):
        preface_string = 'urban_plume_wc_'
    elif f.startswith('urban_plume_'):
        preface_string = 'urban_plume_'
    else:
        try:
            idx = partmc_output_dir[(partmc_output_dir.find('/')+1):].find('/')
            prefix_str = partmc_output_dir[(partmc_output_dir.find('/')+1):][:idx] + '_'
        except:
            try:
                preface_string, repeat2, timestep2 = f.split('_')
                preface_string += '_'
            except:
                preface_string = 'YOU_NEED_TO_CHANGE_preface_string_'
    ncfile = partmc_output_dir / (preface_string + str(int(repeat)).zfill(4) + '_' + str(int(timestep)).zfill(8) + '.nc')
    return ncfile