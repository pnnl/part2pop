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
        insoluble_radius = self.get_insoluble_radius()
        self.INSA = 4.0*np.pi*insoluble_radius**2 # m^2
        for ii, (species) in enumerate(self.base_particle.species):
            spec_modifications=dict(spec_mod.get(species.name, {}))
            try:
                m_Jhet, b_Jhet = retrieve_Jhet_val(species.name, spec_modifications=spec_modifications)
            except:
                m_Jhet = np.nan
                b_Jhet = np.nan
            self.m_log10_Jhet[ii]=m_Jhet
            self.b_log10_Jhet[ii]=b_Jhet
        
    def get_Jhet(self, T):
        vks = np.zeros((len(self.base_particle.species), len(T)))
        spec_Jhets = np.zeros((len(self.base_particle.species), len(T)))
        P_wv, P_ice = calculate_Psat(T)
        aw_ice = P_ice/P_wv
        aw = self.get_aw()
        delta_aw = aw - aw_ice
        for ii, (species, m, b) in enumerate(zip(self.base_particle.species, self.m_log10_Jhet, self.b_log10_Jhet)):
            spec_Jhets[ii] = 10**(m * delta_aw + b)
            vks[ii] = self.base_particle.get_spec_vol(species.name)[0]
        vks=np.array(vks)
        spec_Jhets=np.array(spec_Jhets)
        mask = ~np.isnan(spec_Jhets)
        weighted_sum = np.nansum(spec_Jhets * vks, axis=0)
        weight_sum = np.sum(vks * mask, axis=0)
        return weighted_sum / weight_sum
    
    def get_Jhom(self, T):
        """ Homogeneous ice nucleation rate following Koop et al. 2000 """
        P_wv, P_ice = calculate_Psat(T)
        aw_ice = P_ice/P_wv
        aw = self.get_aw()
        delta_aw = aw - aw_ice
        if delta_aw >= 0.26 and delta_aw <= 0.34:
            Jhom = 10**(
                    -906.7
                    + 8502 * delta_aw
                    + -26924 * delta_aw**2
                    + 29180 * delta_aw**3
                    + -1.522
                )
        elif delta_aw < 0.26:
            Jhom = 0
        elif delta_aw > 0.34:
            delta_aw = 0.34
            Jhom = 10**(
                    -906.7
                    + 8502 * delta_aw
                    + -26924 * delta_aw**2
                    + 29180 * delta_aw**3
                    + -1.522
                )
        return Jhom
    
    def get_aw(self):
        """
        Estimate water activity (aw) from dissolved ion masses using an ideal osmotic coefficient (phi=1):
            ln(aw) = -Mw * sum_i b_i
        where b_i are ion molalities (mol/kg water).

        Returns
        -------
        aw : float
            Water activity (0 < aw <= 1)
        """
        water_mass = self.base_particle.get_spec_mass("H2O")
        if water_mass <= 0:
            raise ValueError("water mass must be > 0")

        # sum molalities of ions
        sum_b = 0.0
        for species, mass in zip(self.base_particle.species, self.base_particle.masses):
            if species.name in ['SO4', 'NO3', 'NH4', 'Cl', 'CO3', 'Na']:
                n_mol = mass / species.molar_mass
                b_molal = n_mol / water_mass # mol/kg
                sum_b += b_molal
        ln_aw = -0.01801528 * sum_b
        aw = np.exp(ln_aw)
        
        # numerical safety
        if aw > 1.0:
            aw = 1.0
        if aw <= 0.0:
            aw = 0.0

        return aw
    
    def get_insoluble_radius(self):
        vks = []
        for ii, (species) in enumerate(self.base_particle.species):
            try:
                m_Jhet_val, b_Jhet_val = retrieve_Jhet_val(species.name)
                vks.append(self.base_particle.get_spec_vol(species.name)[0])
            except:
                continue
        return ((3.0*np.sum(np.array(vks)))/(4.0*np.pi))**(1/3)
    
        

def build(base_particle, config):
    """Optional fallback factory callable for discovery."""
    return HomogeneousParticle(base_particle, config)
