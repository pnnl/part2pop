import numpy as np

from part2pop import species
from ..base import FreezingParticle, retrieve_Jhet_val
from .registry import register
from ...aerosol_particle import Particle
from .utils import calculate_Psat
import warnings

@register("homogeneous")
class HomogeneousParticle(FreezingParticle):
    """
    Homogeneous sphere morphology freezing particle model.

    Constructor expects (base_particle, config) to align with the factory builder.

    Config options:

    """

    def __init__(self, base_particle, config):
        # Initialize as a Particle using the base particle's composition
        super().__init__(base_particle, config)
        spec_mod = dict(config.get("species_modifications", {}))
        specdata_path = config.get("specdata_path", None)
        self.base_particle = base_particle
        self.m_log10_Jhet = np.zeros(self.base_particle.masses.shape)
        self.b_log10_Jhet = np.zeros(self.base_particle.masses.shape)
        insoluble_radius = self.get_insoluble_radius(spec_modifications=spec_mod, specdata_path=specdata_path)
        self.INSA = 4.0*np.pi*insoluble_radius**2 # m^2
        for ii, (species) in enumerate(self.base_particle.species):
            spec_modifications=dict(spec_mod.get(species.name, {}))
            try:
                m_Jhet, b_Jhet = retrieve_Jhet_val(species.name, spec_modifications=spec_modifications, specdata_path=specdata_path)
            except:
                m_Jhet = spec_modifications.get("m_log10_Jhet", np.nan)
                b_Jhet = spec_modifications.get("b_log10_Jhet", np.nan)
                if np.isnan(m_Jhet) and not np.isnan(b_Jhet):
                    warnings.warn(f"Species {species.name} has supplied b_log10_Jhet but missing m_log10_Jhet; setting m_log10_Jhet to NaN.", UserWarning)
                if np.isnan(b_Jhet) and not np.isnan(m_Jhet):
                    warnings.warn(f"Species {species.name} has supplied m_log10_Jhet but missing b_log10_Jhet; setting b_log10_Jhet to NaN.", UserWarning)
            
            self.m_log10_Jhet[ii]=m_Jhet
            self.b_log10_Jhet[ii]=b_Jhet
        
                
    def compute_Jhet(self, T):
        water_mass = self.base_particle.get_spec_mass("H2O")
        if water_mass <= 0:
            return 0.0
        else:
            vks = np.zeros((len(self.base_particle.species)))
            spec_Jhets = np.zeros((len(self.base_particle.species)))
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
            
            # no insoluble species, so Jhet=0.0
            if (mask==False).all():
                return 0.0
            # at least one insoluble species
            else:
                weighted_sum = np.nansum(spec_Jhets * vks, axis=0)
                weight_sum = np.sum(vks * mask, axis=0)
                return weighted_sum / weight_sum
    
    def compute_Jhom(self, T):
        """ Homogeneous ice nucleation rate following Koop et al. 2000 """
        
        water_mass = self.base_particle.get_spec_mass("H2O")
        if len(water_mass)>1:
            raise ValueError("Expected water mass to be a single value, got multiple.")
        else:
            water_mass = water_mass.item()
        if water_mass <= 0:
            return 0.0
        else:
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
            return 1e6*Jhom # m^-3 s^-1
    
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
        # sum molalities of ions
        sum_b = 0.0
        water_mass = self.base_particle.get_spec_mass("H2O")
        if len(water_mass)>1:
            raise ValueError("Expected water mass to be a single value, got multiple.")
        else:
            water_mass = water_mass.item()
        for species, mass, m_Jhet, b_Jhet in zip(self.base_particle.species, self.base_particle.masses, self.m_log10_Jhet, self.b_log10_Jhet):
            if np.isnan(m_Jhet) and np.isnan(b_Jhet) and not species.name == "H2O":
                n_mol = mass / species.molar_mass
                b_molal = n_mol / water_mass # mol/kg
                sum_b += b_molal
        ln_aw = -0.01801528 * sum_b
        aw=np.exp(ln_aw)
        return aw
    
    def get_insoluble_radius(self, spec_modifications={}, specdata_path='species_data'):
        vks = []
        for species in self.base_particle.species:
            try:
                spec_mod = dict(spec_modifications.get(species.name, {}))
                _, _ = retrieve_Jhet_val(species.name, spec_modifications=spec_mod, specdata_path=specdata_path)
                vks.append(self.base_particle.get_spec_vol(species.name)[0])
            except:
                continue
        return ((3.0*np.sum(np.array(vks)))/(4.0*np.pi))**(1/3)
    
        

def build(base_particle, config):
    """Optional fallback factory callable for discovery."""
    return HomogeneousParticle(base_particle, config)
