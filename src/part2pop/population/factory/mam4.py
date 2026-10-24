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
import os
import re
from pathlib import Path
from .registry import register
from ...constants import MOLAR_MASS_DRY_AIR, R, DENSITY_LIQUID_WATER
from ...utilities import power_moments_from_lognormal
from .binned_lognormals import build as build_binned_lognormals
try:
    import netCDF4
    _HAS_NETCDF4 = True
except ModuleNotFoundError:
    netCDF4 = None
    _HAS_NETCDF4 = False
if _HAS_NETCDF4:
    _MAM4_AEROSOL_SPECIES = ("so4", "pom", "soa", "bc", "dst", "ncl")

    def _read_namelist_values(filename):
        """Read scalar ``name = value,`` entries from a MAM4 namelist."""
        values = {}
        with open(filename, "r") as stream:
            for line in stream:
                line = line.split("!", 1)[0]
                match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,/]*)", line)
                if match:
                    raw = match.group(2).strip()
                    try:
                        values[match.group(1).lower()] = float(raw.replace("d", "e").replace("D", "E"))
                    except ValueError:
                        continue
        return values

    def _coagulation_only_pom_fractions(mam_input, numc):
        """Return conserved SO4/POM fractions, or raise if they are unsafe."""
        values = _read_namelist_values(mam_input)
        required_flags = ("mdo_coag", "mdo_gaschem", "mdo_gasaerexch", "mdo_newnuc")
        if any(key not in values for key in required_flags):
            raise ValueError(
                "Cannot infer conserved POM composition: MAM4 process flags are missing"
            )
        if not (
            values["mdo_coag"] == 1
            and values["mdo_gaschem"] == 0
            and values["mdo_gasaerexch"] == 0
            and values["mdo_newnuc"] == 0
        ):
            raise ValueError(
                "Cannot infer conserved POM composition unless MAM4 is coagulation-only"
            )

        mode_fracs = []
        for mode_idx, mode_num in enumerate(numc, start=1):
            raw = np.array([
                values.get(f"mf{species}{mode_idx}", 0.0)
                for species in _MAM4_AEROSOL_SPECIES
            ], dtype=float)
            if mode_num <= 1e-20 or np.sum(raw) <= 1e-20:
                mode_fracs.append(None)
                continue
            unsupported = raw[2:]
            if np.any(np.abs(unsupported) > 1e-14):
                raise ValueError(
                    "Cannot infer conserved POM composition when populated MAM4 "
                    "modes contain species other than SO4 and POM"
                )
            dry_total = raw[0] + raw[1]
            if dry_total <= 0 or raw[0] <= 0:
                raise ValueError("Coagulation-only POM inference requires positive SO4")
            mode_fracs.append(raw[:2] / dry_total)

        populated = [frac for frac in mode_fracs if frac is not None]
        if not populated:
            raise ValueError("Cannot infer conserved POM composition: no populated modes")
        reference = populated[0]
        if any(not np.allclose(frac, reference, rtol=0, atol=1e-12) for frac in populated[1:]):
            raise ValueError(
                "Cannot infer conserved POM composition when populated MAM4 modes "
                "have different initial SO4/POM ratios"
            )
        return np.array([reference if frac is None else frac for frac in mode_fracs])

    @register("mam4") # only registers if netCDF4 is available
    def build(config):
        mam4_dir = Path(config['mam4_dir'])
        timestep = config['timestep']
        GSDs = config['GSD']
        if config.get("D_is_wet", True):
            aero_spec_names = [['SO4','OC','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O']]
        else:
            aero_spec_names = [['SO4','OC'],['SO4','OC'],['SO4','OC'],['SO4','OC']]
        
        N_bins = config['N_bins']
        

        if timestep == 0:
            raise ValueError('timestep=0 is invalid. Specify timestep = 1 for initial conditions')
        elif timestep == 1:
            mam_input = mam4_dir / 'namelist'
            Ns = np.zeros([len(GSDs)])
            
            for kk in range(len(Ns)):
                Ns[kk] = get_mam_input(
                    'numc' + str(kk+1),
                    mam_input)
            
            # GMDs = np.zeros([len(GSDs)])
            # try:
            #     for kk in range(len(GMDs)):
            #         GMDs[kk] = get_mam_input(
            #             'dgn' + str(kk+1),
            #             mam_input)
            # except ValueError:
                # if dgn not in namelist, use default initial GMDs
            GMDs = config.get('GMD_init', [1.1e-7, 2.6e-8, 2e-6, 5e-8]) # default initial GMDs for MAM4 (check fresh BC mode)
            
            aero_spec_fracs = []
            for kk in range(len(Ns)):
                spec_frac = []
                total_frac = 0.
                for spec_name in aero_spec_names[kk]:
                    # fixme: come up with more elegant way to handle PartMC --> MAM4 species name mapping
                    if spec_name.lower() == 'oc':
                        spec_name = 'pom'
                    
                    if spec_name.lower() == 'h2o':
                        frac = 0. 
                        #raise Warning('H2O mass fraction cannot be specified in MAM4 namelist; it is computed internally based on other species and hygroscopic growth. Setting frac_H2O=0 here.')
                    else: 
                        frac = get_mam_input(
                            'mf' + spec_name.lower() + str(kk+1),
                            mam_input)
                                        
                    spec_frac.append(frac)
                    total_frac += frac
                spec_frac = np.array(spec_frac)
                spec_frac /= total_frac
                aero_spec_fracs.append(spec_frac)
            aero_spec_fracs = np.array(aero_spec_fracs)

            lognormals_cfg = {
                'type': 'binned_lognormals',
                'N_sigmas': config.get('N_sigmas',6),
                'D_min':config.get('D_min',None), #fixme: option for +/- sigmas
                'D_max':config.get('D_max',None),
                'N_bins': N_bins,
                'N': Ns,
                'GMD': GMDs,
                'GSD': GSDs,
                'aero_spec_names': aero_spec_names,
                'aero_spec_fracs': aero_spec_fracs,
                }
        else:
            output_tt_idx = timestep - 2 # timestep=2 is output index 0 = after 1 time step

            # fixme: make this +/- certain number of sigmas (rather than min/max diams)
            
            N_bins = config['N_bins']
            species_modifications = config.get('species_modification',{}) # modify species properties during post-processing
            output_filename = mam4_dir / 'mam_output.nc'
            currnc = netCDF4.Dataset(output_filename)

            # Copy the selected record before closing the file.  Keeping all
            # fields on the same mode/time index is important: MAM4's modal
            # diagnostics are records, not independently shifted arrays.
            num_aer = np.asarray(currnc['num_aer'][:, output_tt_idx])
            so4_aer = np.asarray(currnc['so4_aer'][:, output_tt_idx])
            soa_aer = np.asarray(currnc['soa_aer'][:, output_tt_idx])
            dgn_a = np.asarray(currnc['dgn_a'][:, output_tt_idx])
            dgn_awet = np.asarray(currnc['dgn_awet'][:, output_tt_idx])
            currnc.close()

            p = config['p']
            T = config['T']
            #from .binned_lognormals import build as build_binned_lognormals
            
            lognormals_cfg = config

            # fixme: make this right
            rho_dry_air = MOLAR_MASS_DRY_AIR * p / (R * T)
            Ns = num_aer * rho_dry_air # N/m^3
            idx, = np.where(Ns>0)
            Ns = Ns[idx]
            D_is_wet = config.get("D_is_wet", True)
            GMDs_wet = dgn_awet[idx]
            GMDs = dgn_a[idx]
            
            mass_so4 = so4_aer[idx] * rho_dry_air
            mass_soa = soa_aer[idx] * rho_dry_air
            mam_input = mam4_dir / "namelist"

            # MAM4 does not expose POM in mam_output.nc.  In a strictly
            # coagulation-only run, a common initial SO4/POM ratio is conserved,
            # so recover the missing species fraction from the namelist.  This
            # changes composition only; particle masses remain diameter-derived.
            namelist_values = _read_namelist_values(mam_input) if mam_input.exists() else {}
            numc_initial = np.array([
                namelist_values.get(f"numc{mode_idx}", 0.0)
                for mode_idx in range(1, len(GSDs) + 1)
            ])
            initial_pom = np.array([
                namelist_values.get(f"mfpom{mode_idx}", 0.0)
                for mode_idx in range(1, len(GSDs) + 1)
            ])
            if mam_input.exists() and np.any((numc_initial > 1e-20) & (initial_pom > 1e-20)):
                conserved_dry_fracs = _coagulation_only_pom_fractions(mam_input, numc_initial)
                if np.max(np.abs(mass_soa)) > 1e-10 * max(np.max(np.abs(mass_so4)), 1e-30):
                    raise ValueError(
                        "MAM4 output contains SOA in a reconstruction requiring conserved SO4/POM"
                    )
            else:
                conserved_dry_fracs = None
            
            # fixme: not actually sure what assumptions go into this for MAM4, assuming same GSD to compute mass H2O
            mass_h2o = DENSITY_LIQUID_WATER * np.pi/6 * np.array(
                    [
                        (power_moments_from_lognormal(3,N,gmd_wet,gsd) 
                        - power_moments_from_lognormal(3, N, gmd, gsd)) 
                        for (N, gmd, gmd_wet, gsd) in zip(Ns, GMDs, GMDs_wet, GSDs)])
            if D_is_wet:
                aero_spec_names = [['SO4','OC','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O']]
                aero_spec_fracs = []
                for mode_idx, (m_so4,m_soa,m_h2o) in enumerate(zip(mass_so4,mass_soa,mass_h2o)):
            
                    # temporary patch to avoid 0 mass fraction for empty modes
                    # excluding water in check to avoid zero mass for dry radius calculations
                    if m_so4 + m_soa <= 0:
                        m_so4 = 1

                    dry_frac = (
                        conserved_dry_fracs[idx[mode_idx]]
                        if conserved_dry_fracs is not None
                        else np.array([m_so4, m_soa]) / max(m_so4 + m_soa, 1e-300)
                    )
                    spec_frac = np.array([dry_frac[0], dry_frac[1], m_h2o])
                    spec_frac /= np.sum(spec_frac)
                    spec_frac[np.isnan(spec_frac)] = 0.
                    aero_spec_fracs.append(spec_frac)
            else:
                aero_spec_names = [['SO4','OC'],['SO4','OC'],['SO4','OC'],['SO4','OC']]
                aero_spec_fracs = []
                for mode_idx, (m_so4,m_soa) in enumerate(zip(mass_so4,mass_soa)):
            
                    # temporary patch to avoid 0 mass fraction for empty modes
                    if m_so4 + m_soa <= 0:
                        m_so4 = 1

                    spec_frac = (
                        conserved_dry_fracs[idx[mode_idx]]
                        if conserved_dry_fracs is not None
                        else np.array([m_so4, m_soa]) / max(m_so4 + m_soa, 1e-300)
                    )
                    spec_frac /= np.sum(spec_frac)
                    spec_frac[np.isnan(spec_frac)] = 0.
                    aero_spec_fracs.append(spec_frac)

            if D_is_wet:
                GMDs_cfg = GMDs_wet
            else:
                GMDs_cfg = GMDs
            lognormals_cfg = {
                "type": "binned_lognormals",
                "N": Ns,
                "N_sigmas": config.get('N_sigmas', 6),
                "D_is_wet": D_is_wet,
                "GMD": GMDs_cfg,
                "GSD": GSDs,
                "aero_spec_names": aero_spec_names,
                "aero_spec_fracs": aero_spec_fracs,
                "N_bins": N_bins,
                "D_min": config.get('D_min', None),
                "D_max": config.get('D_max', None),
            }
        mam4_population = build_binned_lognormals(lognormals_cfg)
        return mam4_population

else:
    def build(config):
        raise ModuleNotFoundError(
            "Install netCDF4 to read MAM4 files: generate the environment-partmc.yml file using tools/create_conda_env.py, "
            "then create and activate the 'part2pop' conda environment (conda env create -f environment-partmc.yml -n part2pop)."
        )

# fimxe: better way to deal with MAM input? send in namelist (or path to namelist)
# import f90nml
# namelist = f90nml.read('namelist')
# qso2 = namelist['chem_input']['qso2']
def get_mam_input(varname,mam_input_filename):

    f_input = open(mam_input_filename,'r')
    input_lines = f_input.readlines()
    yep = 0
    for oneline in input_lines:
        if varname in oneline:
            yep += 1 
            idx1,=np.where([hi == '=' for hi in oneline])
            idx2,=np.where([hi == ',' for hi in oneline])
            vardat = float(''.join([oneline[ii] for ii in range(idx1[0]+1,idx2[0])]))
    if yep == 0:
        # raise ValueError(varname,'is not a MAM input parameter')
        vardat = 0.0
        print(varname,'is not a MAM input parameter; returning 0.0')    
    elif yep > 1:
        raise ValueError('more than one line in ', mam_input_filename, 'starts with', varname)
    return vardat
