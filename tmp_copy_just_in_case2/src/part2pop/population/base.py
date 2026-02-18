# src/part2pop/population/base.py

from dataclasses import dataclass, field
from typing import Tuple, Dict
from warnings import warn
import numpy as np

from part2pop import Particle
from part2pop import AerosolSpecies

@dataclass
class ParticlePopulation:
    """ParticlePopulation: the definition of a population of particles
    in terms of the number concentrations of different particles """

    species: Tuple[AerosolSpecies, ...] # shape = N_species
    spec_masses: np.array # shape = (N_particles, N_species)
    num_concs: np.array # shape = N_particles
    ids: Tuple[int, ...] # shape = N_particles
    # Population-level species modifications (e.g., density, kappa overrides)
    species_modifications: Dict[str, dict] = field(default_factory=dict)

    def find_particle(self, part_id):
        if part_id in self.ids:
            idx, = np.where([one_id == part_id for one_id in self.ids])
            if len(idx)>1:
                ValueError('part_id is listed more than once in self.ids')
            else:
                idx = idx[0]
        else:
            idx = len(self.ids)
        return idx

    def get_particle(self, part_id):
        if part_id in self.ids:
            idx_particle = self.find_particle(part_id)
            return Particle(self.species, self.spec_masses[idx_particle,:])
        else:
            raise ValueError(str(part_id) + ' not in ids')

    def set_particle(self, particle, part_id, num_conc, suppress_warning=True):
        part_id = int(part_id)
        if part_id not in self.ids:
            if not suppress_warning:
                warn('part_id not in self.ids, adding ' + str(part_id))
            self.add_particle(particle, part_id, num_conc)
        else:
            self.species = particle.species
            idx = self.find_particle(part_id)
            self.spec_masses[idx,:] = particle.masses
            self.num_concs[idx] = num_conc
            self.ids[idx] = part_id

    def add_particle(self, particle, part_id, num_conc):
        if len(self.ids) == 0:
            self.species = particle.species
            self.spec_masses = np.zeros([1,len(particle.species)])
            self.spec_masses[0,:] = particle.masses
            self.num_concs = np.hstack([num_conc])
            self.ids = [part_id]
        else:
            self.spec_masses = np.vstack([self.spec_masses, particle.masses.reshape(1,-1)])
            self.num_concs = np.hstack([self.num_concs, num_conc])
            self.ids.append(part_id)
    
    def get_species_idx(self, spec_name):
        idx, = np.where([
            spec.name in spec_name for spec in self.species])
        return idx[0]
    
    def _equilibrate_h2o(self,S,T,rho_h2o=1000., MW_h2o=18e-3):
        for (part_id,num_conc) in zip(self.ids,self.num_concs):
            particle = self.get_particle(part_id)
            particle._equilibrate_h2o(S,T)
            self.set_particle(particle, part_id, num_conc)
    # fixme: add "equilibrate" for other species, too.

    def get_effective_radius(self):
        rs = []
        for part_id in self.ids:
            particle = self.get_particle(part_id)
            rs.append(particle.get_Dwet()/2.)
        rs = np.asarray(rs)
        Ns = self.num_concs
        return np.sum(rs*Ns)/np.sum(Ns)

    def get_Ntot(self):
        return np.sum(self.num_concs)
    
    def get_particle_var(self, varname, *kwargs):
        return np.array([self.get_particle(part_id).get_variable(varname, *kwargs) for part_id in self.ids])
    
    def get_num_dist_1d(
            self, varname = 'wet_diameter', 
            density=True, weights=None, method='hist', 
            N_bins=20, x_range=None, *kwargs):
       
        vardat = self.get_particle_var(varname, *kwargs)
        if method == 'hist':
            return np.histogram(vardat, bins=N_bins, range=x_range, density=density, weights=weights)
        else:
            raise NotImplementedError(f"{method} not yet implemented")
    
    def get_tot_mass(self):
        return np.sum(self.num_concs*np.sum(self.spec_masses,axis=1))
    
    def get_tot_dry_mass(self):
        idx_h2o, = np.where([spec.name == 'H2O' for spec in self.species])
        if len(idx_h2o)==0:
            return self.get_tot_mass()
        else:
            return np.sum(self.num_concs*np.sum(self.spec_masses[:,np.arange(len(self.species))!=idx_h2o[0]],axis=1))
        
    def get_mass_conc(self,spec_name):
        idx, = np.where([spec.name == spec_name for spec in self.species])
        return np.sum(self.num_concs*self.spec_masses[:,idx[0]])
    
    # FIXME: double-check this method
    def reduce_mixing_state(self, mixing_state='part_res', 
                            RH=None, T=None, 
                            sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
        idx_bc, = np.where([spec.name.upper() == 'BC' for spec in self.species])
        idx_not_h2o, = np.where([spec.name != 'H2O' for spec in self.species])

        if mixing_state.startswith('MAM5'):
            avg_these = np.where(self.spec_masses[:,idx_bc]>0)
        elif mixing_state.startswith('MAM4'):
            avg_these = np.arange(self.spec_masses.shape[0]) 
        elif mixing_state == 'part_res':
            avg_these = np.array([]) # unit test to make sure nothing happens
        
        if mixing_state.endswith('sameDryMass'):
            mixing_factor = np.sum(self.spec_masses[idx_not_h2o,avg_these],axis=1)/np.sum(np.sum(self.spec_masses[idx_not_h2o,avg_these]))
            normalized_by = np.sum(self.spec_masses[idx_not_h2o,avg_these],axis=0)
        elif mixing_state.endswith('sameBC'):
            mixing_factor = np.sum(self.spec_masses[idx_not_h2o,avg_these],axis=1)/np.sum(self.spec_masses[idx_bc,avg_these],axis=1)
            normalized_by = self.spec_masses[idx_bc,avg_these]

        for ii,part_id in enumerate(self.ids[avg_these]):
            particle = self.get_particle(part_id)
            particle.spec_masses[idx_not_h2o] = mixing_factor*normalized_by[ii]
            Dwet = particle.get_Dwet(RH=RH, T=T, sigma_h2o=sigma_h2o, rho_h2o=rho_h2o, MW_h2o=MW_h2o)
            Ddry = particle.get_Ddry()
            vol_h2o = np.pi/6.*(Dwet**3 - Ddry**3)
            mass_h2o = vol_h2o*rho_h2o
            particle.spec_masses[particle.idx_h2o()] = mass_h2o
            num_conc =self.num_conc[ii]
            self.set_particle(particle, part_id, num_conc, suppress_warning=False)