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


class FreezingParticle(Particle):
    """
    Base class for all freezing particle morphologies.
    """
    
    @abstractmethod
    def compute_Jhet(self, T):
        """Compute per-particle heterogeneous ice nucleation rate.
        """

    

class FreezingPopulation(ParticlePopulation):
    """
    Manages a population of freezing particles, possibly of mixed morphologies.
    Holds cross-section cubes per particle and provides population-aggregated optics.
    """

    def __init__(self, base_population, T_grid):
        # Initialize ParticlePopulation state
        super().__init__(
            species=base_population.species,
            spec_masses=np.array(base_population.spec_masses, copy=True),
            num_concs=np.array(base_population.num_concs, copy=True),
            ids=list(base_population.ids).copy(),
        )

        # Prepare storage for per-particle Jhet values
        N_part = len(self.ids)
        self.T_grid = T_grid
        self.Jhet = np.zeros((len(T_grid), N_part), dtype=float)
        self.INSA = np.zeros((len(T_grid), N_part), dtype=float)
    
    def add_freezing_particle(self, freezing_particle, part_id, T, **kwargs):
        idx = self.find_particle(part_id)
        if idx >= len(self.ids) or self.ids[idx] != part_id:
            raise ValueError(f"part_id {part_id} not found in OpticalPopulation ids.")
        self.Jhet[:,idx] = freezing_particle.get_Jhet(T)
        self.INSA[:,idx] = freezing_particle.INSA
    
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
        
        
    
    def get_freezing_probs(self):
        return 1-np.exp(-self.Jhet*self.INSA*1.0)

        


def retrieve_Jhet_val(name, spec_modifications={}):
    # 'specdata_path' kept for backwards compatibility but ignored
    
    # todo: do we want to add Jhets to the species? Make "FreezingSpecies" class under base and update building?
    
    with open_dataset('species_data/freezing_data.dat') as fh:
        for line in fh:
            if line.strip().startswith("#"):
                continue
            if line.upper().startswith(name.upper()):
                name_in_file, m_Jhet, b_Jhet = line.split()
                m_Jhet_val = spec_modifications.get('m_log10Jhet', m_Jhet)
                b_Jhet_val = spec_modifications.get('b_log10Jhet', b_Jhet)
    return m_Jhet_val, b_Jhet_val
