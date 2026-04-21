"""Freezing particle base classes.

Defines abstract `FreezingParticle` interface which
aggregates per-particle Jhet and ice nucleating surface area.
"""

from abc import abstractmethod
import numpy as np

# Adjust imports to your tree
from ..aerosol_particle import Particle
from ..population.base import ParticlePopulation
# from ..data_old import species_open
from ..data import open_dataset
from scipy.integrate import trapezoid
from .builder import FreezingParticleBuilder

class FreezingParticle(Particle):
    """
    Base class for all freezing particle morphologies.
    """
    def __init__(self, base_particle, config):
        super().__init__(species=base_particle.species, masses=base_particle.masses)
        species_modifications = config.get("species_modifications", {})
        specdata_path = config.get("specdata_path", None)
        self.m_log10_Jhet = np.zeros(self.masses.shape)
        self.b_log10_Jhet = np.zeros(self.masses.shape)
        self.INSA = 0.0
        self.species_modifications = species_modifications
        self.species_data_path = specdata_path
    
    @abstractmethod
    def compute_Jhet(self, T):
        """Returns array of Jhet values for each particle in the population."""
    
    @abstractmethod
    def compute_Jhom(self, T):
        """Returns array of Jhom values for each particle in the population."""

class FreezingPopulation(ParticlePopulation):
    """
    Manages a population of freezing particles, possibly of mixed morphologies.
    Holds cross-section cubes per particle and provides population-aggregated optics.
    """

    def __init__(self, base_population):
        # Initialize ParticlePopulation state
        super().__init__(
            species=base_population.species,
            spec_masses=np.array(base_population.spec_masses, copy=True),
            num_concs=np.array(base_population.num_concs, copy=True),
            ids=list(base_population.ids).copy(),
            species_modifications=base_population.species_modifications.copy()
        )

        # Prepare storage for per-particle Jhet values
        N_part = len(self.ids)
        self.m_log10_Jhet = np.zeros((N_part, len(base_population.species)), dtype=float)
        self.b_log10_Jhet = np.zeros((N_part, len(base_population.species)), dtype=float)
        self.INSA = np.zeros(N_part, dtype=float) # m^2
    
    def compute_Jhets(self, T, config):
        return np.array([self.get_freezing_particle(part_id, config).compute_Jhet(T) for part_id in self.ids])
    
    def compute_Jhoms(self, T, config):
        return np.array([self.get_freezing_particle(part_id, config).compute_Jhom(T) for part_id in self.ids])
        
    # def find_particle(self, part_id):
    #     if part_id in self.ids:
    #         idx, = np.where([one_id == part_id for one_id in self.ids])
    #         if len(idx)>1:
    #             ValueError('part_id is listed more than once in self.ids')
    #         else:
    #             idx = idx[0]
    #     else:
    #         idx = len(self.ids)
    #     return idx
        
    def get_freezing_particle(self, part_id, config):
        if part_id in self.ids:
            idx_particle = self.find_particle(part_id)
            base_particle = Particle(self.species, self.spec_masses[idx_particle])
            return FreezingParticleBuilder(config).build(base_particle)
        else:
            raise ValueError(str(part_id) + ' not in ids')
        
    def add_freezing_particle(self, freezing_particle, part_id):
        idx = self.find_particle(part_id)
        if idx >= len(self.ids) or self.ids[idx] != part_id:
            raise ValueError(f"part_id {part_id} not found in FreezingPopulation ids.")
        self.m_log10_Jhet[idx,:] = freezing_particle.m_log10_Jhet
        self.b_log10_Jhet[idx,:] = freezing_particle.b_log10_Jhet
        self.INSA[idx] = freezing_particle.INSA
    
    def get_avg_Jhet(self):
        weights = np.tile(self.num_concs, (len(self.T_grid), 1))
        return np.average(self.Jhet, weights=weights, axis=1)
    
    def get_nucleating_sites(self, dT_dt):
        out = np.zeros(self.T_grid.shape)
        if self.T_grid[-1]>self.T_grid[0]:
            for ii in range(1, len(self.T_grid)+1):
                out[-ii] = np.sum((self.num_concs/dT_dt)*trapezoid(np.flip(self.Jhet[-ii:]), x=np.flip(self.T_grid[-ii:]), axis=0))
        else: 
            for ii in range(0, len(self.T_grid)):
                out[ii] = np.sum((self.num_concs/dT_dt)*trapezoid(self.Jhet[:ii], x=self.T_grid[:ii], axis=0))       
        return out
    
    def get_frozen_fraction(self, dT_dt):
        out = np.zeros(self.T_grid.shape)
        weights = self.num_concs/np.sum(self.num_concs)
        if self.T_grid[-1]>self.T_grid[0]:
            for ii in range(1, len(self.T_grid)+1):
                ns = (1/dT_dt)*trapezoid(np.flip(self.Jhet[-ii:]), x=np.flip(self.T_grid[-ii:]), axis=0)
                out[-ii]=1-np.sum(weights*np.exp(-1.0*ns*self.INSA[-ii]))
        else: 
            for ii in range(0, len(self.T_grid)):
                ns = (1/dT_dt)*trapezoid(self.Jhet[:ii], x=self.T_grid[:ii], axis=0)
                out[ii]=1-np.sum(weights*np.exp(-1.0*ns*self.INSA[ii]))
        return out
    
    def get_species_idx(self, spec_name):
        names = [spec.name for spec in self.species]
        try:
            return np.where(np.array(names) == spec_name)[0][0]
        except:
            return None
    
    def get_freezing_probs(self, dt=1.0):
        water_volumes = self.spec_masses[:,self.get_species_idx("H2O")]/self.species[self.get_species_idx("H2O")].density
        P_frz = 1-np.exp(-(self.Jhet*self.INSA + self.Jhom*water_volumes)*dt)
        return P_frz


def retrieve_Jhet_val(name, spec_modifications={}, specdata_path='species_data'):
    # 'specdata_path' kept for backwards compatibility but ignored
    # todo: do we want to add Jhets to the species? Make "FreezingSpecies" class under base and update building?    
    if specdata_path is None:
        specdata_path = 'species_data'

    with open_dataset(specdata_path+'/freezing_data.dat') as fh:
        for line in fh:
            if line.strip().startswith("#"):
                continue
            if line.upper().startswith(name.upper()):
                _, m_Jhet, b_Jhet = line.split()
                m_Jhet_val = spec_modifications.get('m_log10_Jhet', m_Jhet)
                b_Jhet_val = spec_modifications.get('b_log10_Jhet', b_Jhet)
           
    return m_Jhet_val, b_Jhet_val
    
