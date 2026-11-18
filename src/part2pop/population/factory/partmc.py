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
import re
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


# PartMC names its snapshots <run name>_<repeat>_<timestep>.nc. The run name can
# be anything, including names that contain underscores.
_NCFILE_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<repeat>\d+)_(?P<timestep>\d+)\.nc$")


def get_ncfile(partmc_output_dir, timestep, repeat):
    """Return the PartMC output file for one repeat and output timestep.

    Files are matched by name, so any run name works and other files in the folder
    (a macOS ``.DS_Store``, ``._*`` copy artifacts, notes) are ignored.
    """
    partmc_output_dir = Path(partmc_output_dir)
    if not partmc_output_dir.exists():
        raise FileNotFoundError(f"PartMC output directory {partmc_output_dir} does not exist.")
    repeat, timestep = int(repeat), int(timestep)

    matches, timesteps = [], set()
    for path in sorted(partmc_output_dir.glob("*.nc")):
        match = _NCFILE_PATTERN.match(path.name)
        if match is None or path.name.startswith(".") or int(match["repeat"]) != repeat:
            continue
        timesteps.add(int(match["timestep"]))
        if int(match["timestep"]) == timestep:
            matches.append(path)

    if len(matches) > 1:
        raise ValueError(
            f"More than one PartMC file matches repeat {repeat}, timestep {timestep} "
            f"in {partmc_output_dir}: {[p.name for p in matches]}. Keep one run per folder."
        )
    if not matches:
        raise FileNotFoundError(
            f"No PartMC file for repeat {repeat}, timestep {timestep} in {partmc_output_dir}; "
            f"timesteps available for this repeat: {sorted(timesteps)}"
        )
    return matches[0]