import numpy as np
from ..base import FreezingParticle, retrieve_Jhet_val
from .registry import register
from ...aerosol_particle import Particle
from .utils import calculate_Psat

@register("homogeneous")
class HomogeneousParticle(FreezingParticle):
    """
    Homogeneous sphere morphology freezing particle model.

    Constructor expects (base_particle, config) to align with the factory builder.

    Config options:

    """

    def __init__(self, base_particle, config):
        # Initialize as a Particle using the base particle's composition
        super().__init__(base_particle.species, base_particle.masses)
        
        spec_mod = dict(config.get("species_modifications", {}))
        self.base_particle = base_particle
        self.m_log10_Jhet = np.zeros(self.base_particle.masses.shape)
        self.b_log10_Jhet = np.zeros(self.base_particle.masses.shape)
        Dwet = base_particle.get_Dwet()
        self.INSA = 4.0*np.pi*(Dwet/2)**2 # m^2
        for ii, (species) in enumerate(self.base_particle.species):
            spec_modifications=dict(spec_mod.get(species.name, {}))
            m_Jhet, b_Jhet = retrieve_Jhet_val(species.name, spec_modifications=spec_modifications)
            self.m_log10_Jhet[ii]=m_Jhet
            self.b_log10_Jhet[ii]=b_Jhet
        
    def get_Jhet(self, T):
        vks = []
        spec_Jhets = []
        P_wv, P_ice = calculate_Psat(T)    
        aw_ice = P_ice/P_wv
        aw = 1.0 # pure droplets, will be < 1.0 for solutions
        delta_aw = aw - aw_ice
        for ii, (species, m, b) in enumerate(zip(self.base_particle.species, self.m_log10_Jhet, self.b_log10_Jhet)):
            if species.name != 'H2O':
                spec_Jhets.append(10**(m * delta_aw + b))
                vks.append(self.base_particle.get_spec_vol(species.name)[0])
        return np.average(spec_Jhets, weights=vks, axis=0)
        

def build(base_particle, config):
    """Optional fallback factory callable for discovery."""
    return HomogeneousParticle(base_particle, config)
