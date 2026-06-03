#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from HI-SCALE style observational inputs:
  - FIMS: size distribution (binned N(Dp) or dN/dlnDp / dN/dlog10Dp)
  - AIMMS: aircraft state for altitude/region/time filtering
  - miniSPLAT: reduced-class number fractions
  - AMS: bulk mass fractions (SO4, NO3, OC, NH4)

Design goals:
  - Keep everything in ONE builder file (no separate io module)
  - Follow the structure of other population builders in part2pop
  - Deterministic: split each FIMS size bin into particle "types"
  - Minimal dependencies: numpy only

Config keys (required):
{
  "type": "hiscale_observations",
  "fims_file": "...",
  "fims_bins_file": "...",          # explicit lo/hi bin edges (nm)
  "aimms_file": "...",
  "splat_file": "...",
  "ams_file": "...",
  "z": 100.0,
  "dz": 2.0,
  "splat_species": { "BC": ["BC"], "OIN": ["OIN"], ... }
}

Config keys (optional):
  cloud_flag_value: int/float, default 0
  max_dp_nm: float, default 1000
  splat_cutoff_nm: float, default 0 (no cutoff)
  region_filter: dict with lon/lat bounds (defaults match separate_tools)
  composition_strategy: "ams_everywhere" | "templates_fallback_to_ams" | "templates_only"
  type_templates: dict type->dict(spec->massfrac)
  ams_species_map: dict AMS key-> part2pop species name (default {"SO4":"SO4","NO3":"NO3","OC":"OC","NH4":"NH4"})
  species_modifications: dict spec_name -> property overrides
  D_is_wet: bool, default False

  aimms_time_col/alt/lat/lon: override AIMMS column names if needed
  aimms_cloud_col: optional AIMMS cloud column (usually None)
  fims_time_col: optional FIMS time column name override
  fims_cloud_col: optional FIMS cloud column name override (default "Cloud_flag")

  fims_density_measure: "ln" | "log10" | "per_bin"
      - "ln" means FIMS bins are dN/dlnDp
      - "log10" means dN/dlog10Dp
      - "per_bin" means already per-bin N
"""

from __future__ import annotations
from typing import Any, Dict

import numpy as np

from ..utils import normalize_population_config
from .helpers.assembly import assemble_population_from_mass_fractions
from .helpers import hiscale as hiscale_helpers
from .registry import register






# -----------------------------
# Builder entrypoint
# -----------------------------

@register("hiscale_observations")
def build(config: Dict[str, Any]) -> ParticlePopulation:
    config = normalize_population_config(config)
    required = ["aimms_file", "splat_file", "ams_file", "z", "dz", "splat_species", "mass_thresholds"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"hiscale_observations missing required config keys: {missing}")
    if "fims_file" not in config and "beasd_file" not in config:
        raise KeyError("hiscale_observations requires either 'fims_file' or 'beasd_file' in config.")
    if "fims_file" in config and "fims_bins_file" not in config:
        raise KeyError("hiscale_observations with 'fims_file' requires 'fims_bins_file' in config.")

    if "fims_file" in config:
        fims_file = str(config["fims_file"])
        fims_bins_file = str(config["fims_bins_file"])
        fims_time_col = config.get("fims_time_col", None)      # usually None; auto-detect
        fims_cloud_col = str(config.get("fims_cloud_col", "Cloud_flag"))
        fims_density_measure = str(config.get("fims_density_measure", "ln"))
    elif "beasd_file" in config:
        beasd_file = str(config["beasd_file"])
        beasd_time_col = config.get("beasd_time_col", None)      # usually None; auto-detect
        beasd_cloud_col = str(config.get("beasd_cloud_col", "Cloud_flag"))
        beasd_density_measure = str(config.get("beasd_density_measure", "per_bin"))

    aimms_file = str(config["aimms_file"])
    splat_file = str(config["splat_file"])
    ams_file = str(config["ams_file"])
    N_particles = int(config.get("N_particles", 1000))
    z = float(config["z"])
    dz = float(config["dz"])
    splat_species = dict(config["splat_species"])
    mass_thresholds = dict(config["mass_thresholds"])
    size_dist_grid = int(config.get("size_dist_grid", 3))
    preferred_matching = str(config.get("preferred_matching", "mass")) # can be either mass or number, determines how the number concentration gets scaled
    cloud_flag_value = config.get("cloud_flag_value", 0)
    max_dp_nm = float(config.get("max_dp_nm", 1000.0))
    splat_cutoff_nm = float(config.get("splat_cutoff_nm", 0.0))
    region_filter = config.get("region_filter", None)
    fill_bins = config.get("fill_bins", True)

    composition_strategy = str(config.get("composition_strategy", "ams_everywhere"))
    type_templates = config.get("type_templates", None)
    ams_species_map = dict(config.get("ams_species_map", {"SO4": "SO4", "NO3": "NO3", "OC": "OC", "NH4": "NH4"}))

    species_modifications = config.get("species_modifications", {})
    D_is_wet = bool(config.get("D_is_wet", False))

    aimms_time_col = str(config.get("aimms_time_col", "Time(UTC)"))
    aimms_alt_col = str(config.get("aimms_alt_col", "Alt"))
    aimms_lat_col = str(config.get("aimms_lat_col", "Lat"))
    aimms_lon_col = str(config.get("aimms_lon_col", "Lon"))
    aimms_cloud_col = config.get("aimms_cloud_col", None)  # usually None

    # --- read average size distribution ---
    if "fims_file" in config:
        Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3 = hiscale_helpers._read_fims_avg_size_dist(
            fims_file=fims_file,
            fims_bins_file=fims_bins_file,
            aimms_file=aimms_file,
            z=z,
            dz=dz,
            cloud_flag_value=cloud_flag_value,
            max_dp_nm=max_dp_nm,
            region_filter=region_filter,
            aimms_time_col=aimms_time_col,
            aimms_alt_col=aimms_alt_col,
            aimms_lat_col=aimms_lat_col,
            aimms_lon_col=aimms_lon_col,
            aimms_cloud_col=aimms_cloud_col,
            fims_time_col=fims_time_col,
            fims_cloud_col=fims_cloud_col,
            fims_density_measure=fims_density_measure,
        )
        size_dist_type="FIMS"
        size_dist_file=fims_file
        size_dist_time_col=fims_time_col
        size_dist_cloud_col=fims_cloud_col
        size_dist_density_measure=fims_density_measure
    
    elif "beasd_file" in config:
        Dp_lo_nm, Dp_hi_nm, N_cm3, N_std_cm3 = hiscale_helpers._read_beasd_avg_size_dist(
            beasd_file=beasd_file,
            aimms_file=aimms_file,
            z=z,
            dz=dz,
            cloud_flag_value=cloud_flag_value,
            max_dp_nm=max_dp_nm,
            region_filter=region_filter,
            aimms_time_col=aimms_time_col,
            aimms_alt_col=aimms_alt_col,
            aimms_lat_col=aimms_lat_col,
            aimms_lon_col=aimms_lon_col,
            aimms_cloud_col=aimms_cloud_col,
            beasd_time_col=beasd_time_col,
            beasd_cloud_col=beasd_cloud_col,
            beasd_density_measure=beasd_density_measure,
        )
        size_dist_type="BEASD"
        size_dist_file=beasd_file
        size_dist_time_col=beasd_time_col
        size_dist_cloud_col=beasd_cloud_col
        size_dist_density_measure=beasd_density_measure
    
    Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)

    # --- miniSPLAT reduced-class number fractions ---
    type_fracs, type_fracs_err = hiscale_helpers._read_minisplat_number_fractions(
        splat_file=splat_file,
        aimms_file=aimms_file,
        size_dist_type=size_dist_type,
        size_dist_file=size_dist_file,
        splat_species=splat_species,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=size_dist_time_col,
        size_dist_cloud_col=size_dist_cloud_col,
    )
    tf = hiscale_helpers._normalize_fracs(type_fracs)
    if not tf:
        raise RuntimeError("miniSPLAT type fractions sum to 0; cannot build population.")

    # --- AMS mass fractions + measured total mass (for optional scaling) ---
    ams_mass_frac, ams_mass_frac_err, measured_mass, measured_mass_err = hiscale_helpers._read_ams_mass_fractions(
        ams_file=ams_file,
        aimms_file=aimms_file,
        size_dist_type=size_dist_type,
        size_dist_file=size_dist_file,
        z=z,
        dz=dz,
        cloud_flag_value=cloud_flag_value,
        region_filter=region_filter,
        aimms_time_col=aimms_time_col,
        aimms_alt_col=aimms_alt_col,
        aimms_lat_col=aimms_lat_col,
        aimms_lon_col=aimms_lon_col,
        aimms_cloud_col=aimms_cloud_col,
        size_dist_time_col=size_dist_time_col,
        size_dist_cloud_col=size_dist_cloud_col,
        ams_time_col=config.get("ams_time_col", None),
        ams_flag_col=config.get("ams_flag_col", None),
    )  

    # --- units ---
    # N_cm3 is per-bin number concentration (cm^-3) returned by _read_fims_avg_size_dist.
    # Convert to m^-3 for particle allocation and derive dN/dlnD for diagnostics/metadata.
    Dp_mid_m = Dp_mid_nm * 1e-9
    Dp_lo_m = Dp_lo_nm * 1e-9
    Dp_hi_m = Dp_hi_nm * 1e-9
    N_m3 = N_cm3 * 1e6  # cm^-3 -> m^-3
    dln = np.log(Dp_hi_nm / Dp_lo_nm)
    if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
        raise ValueError("Invalid FIMS bin edges; cannot compute dln widths.")
    dNdln_m3 = N_m3 / dln

    # break the size distribution into N modes and provide fitting parameters
    size_distribution_pars = hiscale_helpers.fit_Nmodal_distribution(Dp_mid_m, N_m3)
    
    # move the splat species into the different modes to optimize matching with the size distribution and measured mass fractions
    mode_fractions, N_multiplier = hiscale_helpers.optimize_splat_species_distributions(
        splat_species=splat_species,
        size_distribution_pars=size_distribution_pars,
        measured_Dp=Dp_mid_m,
        measured_N=N_m3,
        splat_number_fractions=tf,
        ams_mass_fractions=ams_mass_frac,
        splat_cutoff_nm=splat_cutoff_nm,
        mass_thresholds=mass_thresholds,
        datapoints=1000
    )

    # set up checks
    checks=[False]
    counter = 0
    maxcounter=100
    particle_species=list(tf.keys())
    while (sum(checks)!=len(checks) and counter<maxcounter):
        
        for _ in range(100):
            try:
                # sample which particles are which 
                ptypes = []
                for species in particle_species:
                    ptypes.extend([species] * 1)
                remaining = N_particles - len(ptypes)
                if remaining > 0:
                    ptypes.extend(np.random.choice(particle_species, size=remaining).tolist())
                np.random.shuffle(ptypes)

                # sample the type-dependent size and concentration of each particle
                particle_diameters, particle_num_concs = hiscale_helpers.sample_particle_Dp_N(
                    N_particles, ptypes, tf, Dp_mid_m, Dp_lo_m, Dp_hi_m, N_m3, 
                    mode_fractions, size_distribution_pars, splat_cutoff_nm=splat_cutoff_nm,
                    N_multiplier=N_multiplier, fill_bins=fill_bins, size_dist_grid=size_dist_grid)
                break   
            
            except:
                pass
        else:
            raise ValueError(f"Sampling unsuccessful when fill_bins = {fill_bins}.")

        # change number concentrations to match measurements
        mult={}
        for ptype in tf.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==ptype, particle_diameters>=splat_cutoff_nm*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff_nm*1e-9)
            modeled_Nfraction=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
            mult[ptype]=tf[ptype]/modeled_Nfraction
        for ptype in tf.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==ptype, particle_diameters>=splat_cutoff_nm*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff_nm*1e-9)
            particle_num_concs[spec_idx[0]]*=mult[ptype]
            modeled_Nfraction=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])

        # build a map of mass fractions
        species_map = []
        for ptype in mass_thresholds.keys():
            for species in mass_thresholds[ptype][1]:
                species_map.append(species)
        if "SO4" in species_map or "NO3" in species_map:
            if "NH4" not in species_map:
                species_map.append("NH4")
        species_map = np.asarray(species_map)

        # sample the mass fraction of species in each particle
        aero_spec_fracs=np.zeros((len(ptypes), len(species_map)))
        for ii in range(len(ptypes)):
            temp_names, temp_fracs = hiscale_helpers.sample_particle_masses(ptypes[ii], mass_thresholds, rng=None, max_tries=10_000)
            for name, frac in zip(temp_names, temp_fracs):            
                try:
                    jj = np.where(species_map==name)[0][0]
                except:
                    raise ValueError(f"No place in species map for {name}.")
                aero_spec_fracs[ii,jj]=frac
        aero_spec_names = np.tile(species_map, (N_particles, 1))

        # make the particle population from the list of species and mass fractions
        particle_population = assemble_population_from_mass_fractions(
            diameters=particle_diameters,
            number_concentrations=particle_num_concs,
            species_names=aero_spec_names,
            mass_fractions=aero_spec_fracs,
            species_modifications=species_modifications,
            D_is_wet=D_is_wet,
        )

        # check if the bulk mass fraction matches measurements
        sampled_mass_fractions, mass_fraction_checks = hiscale_helpers.mass_fraction_comparison(
            particle_population, ams_mass_frac, ams_mass_frac_err, mass_thresholds)
        
        # check that particles match miniSPLAT measurements
        sampled_number_fractions, number_fraction_checks = hiscale_helpers.number_fraction_comparison(
            particle_population, mass_thresholds, tf, type_fracs_err, splat_cutoff_nm=splat_cutoff_nm)

        # combine the checks
        checks = mass_fraction_checks+number_fraction_checks
        counter += 1
    
    # change the number concentrations so that the total
    # mass concentration matches the AMS measurements or the 
    # total number concentration matches the FIMS/BEASD
    if preferred_matching=="mass":
        total_mass = np.sum(np.sum(particle_population.spec_masses, axis=1)*particle_population.num_concs)
        particle_population.num_concs*=measured_mass/(1e9*total_mass)
    elif preferred_matching=="number":
        particle_population.num_concs*=np.nansum(N_m3)/np.sum(particle_population.num_concs)
    else:
        raise ValueError(f"preferred_matching must be 'number' or 'mass', got {preferred_matching}.")
    
    # Save particle population metadata
    particle_population.metadata = {
        "source": "hiscale_observations",
        "z": z,
        "dz": dz,
        "cloud_flag_value": cloud_flag_value,
        "max_dp_nm": max_dp_nm,
        "splat_cutoff_nm": splat_cutoff_nm,
        "size_distribution_density_measure": size_dist_density_measure,
        "type_fracs": tf,
        "type_fracs_err": type_fracs_err,
        "ams_mass_frac": ams_mass_frac,
        "ams_mass_frac_err": ams_mass_frac_err,
        "measured_ams_total_mass": measured_mass,
        "measured_ams_total_mass_err": measured_mass_err,
        "size_distribution": {
            "Dp_lo_nm": Dp_lo_nm.copy(),
            "Dp_hi_nm": Dp_hi_nm.copy(),
            "dln": dln.copy(),
            "N_bin_m3": N_m3.copy(),
            "dNdlnD_m3": dNdln_m3.copy(),
        },
    }

    return particle_population
