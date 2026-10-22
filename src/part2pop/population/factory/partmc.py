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


def _load_particle_rows(currnc):
    aero_spec_names = currnc.variables["aero_species"].names.split(",")
    spec_masses = np.array(currnc.variables["aero_particle_mass"][:])
    part_ids = np.array([one_id for one_id in currnc.variables["aero_id"][:]], dtype=int)

    if "aero_num_conc" in currnc.variables.keys():
        num_concs = currnc.variables["aero_num_conc"][:]
    else:
        num_concs = 1.0 / currnc.variables["aero_comp_vol"][:]

    return aero_spec_names, spec_masses, part_ids, num_concs


def _select_particle_indices(part_ids, n_particles):
    if n_particles is None:
        return np.arange(len(part_ids))
    if n_particles <= len(part_ids):
        return np.random.choice(np.arange(len(part_ids)), size=n_particles, replace=False)
    raise IndexError("n_particles > len(part_ids)")


def _resolve_target_total_number(num_concs, N_tot):
    if N_tot is None:
        return np.sum(num_concs)
    return N_tot


def _rescale_selected_number_concentrations(num_concs, idx, N_tot):
    return num_concs[idx] * N_tot / np.sum(num_concs[idx])


def _extract_gas_mixing_ratios(currnc):
    return np.array(currnc.variables["gas_mixing_ratio"][:])


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

    aero_spec_names, spec_masses, part_ids, num_concs = _load_particle_rows(currnc)
    # Get AerosolSpecies objects with modifications if any
    species_list = tuple(
        get_species(name, specdata_path, **species_modifications.get(name, {}))
        for name in aero_spec_names
    )

    N_tot = _resolve_target_total_number(num_concs, N_tot)
    idx = _select_particle_indices(part_ids, n_particles)
    num_concs_selected = _rescale_selected_number_concentrations(num_concs, idx, N_tot)

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
        selected_position = np.where(idx == ii)[0][0]
        partmc_population.set_particle(particle, part_ids[ii], num_concs_selected[selected_position], suppress_warning=suppress_warning)

    if add_mixing_ratios:
        gas_mixing_ratios = _extract_gas_mixing_ratios(currnc)
        partmc_population.gas_mixing_ratios = gas_mixing_ratios
    return partmc_population


def map_camp_specs(camp_spec_names):
    spec_names = []
    for spec_name in camp_spec_names:
        split_specs = spec_name.split('.')
        spec_names.append(split_specs[-1])
    return spec_names


def get_ncfile(partmc_output_dir, timestep, repeat):

    """Find exactly one PartMC NetCDF file under the given directory.

    Recursively accept any prefix followed by ``_{repeat:04d}_{timestep:08d}.nc``.
    Raise FileNotFoundError if absent and ValueError if multiple matches exist;
    never infer which duplicate is the intended output.
    """
    partmc_output_dir = Path(partmc_output_dir)
    if not partmc_output_dir.is_dir():
        raise FileNotFoundError(f"PartMC output directory {partmc_output_dir} does not exist.")
    suffix = f"_{int(repeat):04d}_{int(timestep):08d}.nc"
    candidates = sorted(
        p for p in partmc_output_dir.rglob(f"*{suffix}") if p.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No PartMC file ending with {suffix} in {partmc_output_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous PartMC output for {suffix}: {candidates}")
    return candidates[0]
