# src/part2pop/population/base.py

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from warnings import warn
import numpy as np

from part2pop import Particle
from part2pop import AerosolSpecies

def _same_field_value(left, right):
    """Compare species property values, treating two NaN values as equal.

    A plain ``!=`` reports NaN as different from itself, so a species with an
    unset (NaN) property would never match an identical copy of itself.
    """
    if left is right:
        return True
    if left == right:
        return True
    try:
        return bool(np.isnan(left) and np.isnan(right))
    except TypeError:
        return False


@dataclass
class ParticlePopulation:
    """ParticlePopulation: the definition of a population of particles
    in terms of the number concentrations of different particles """

    species: Tuple[AerosolSpecies, ...] # shape = N_species
    spec_masses: np.array # shape = (N_particles, N_species)
    num_concs: np.array # shape = N_particles
    ids: Tuple[int, ...] # shape = N_particles
    classes: Optional[Tuple[str, ...]] = None # shape = N_particles
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

    def _aligned_particle_masses(self, particle):
        """Validate and align an incoming row without changing population state."""
        # Precision scope starts with values already in particle.masses.
        # Particle.__init__ casts to float64, so bits lost there cannot be
        # restored here. The extended-precision tests use particle-like objects
        # to exercise population updates independently of that constructor.
        masses = np.asarray(particle.masses)
        masses = np.asarray(masses, dtype=np.result_type(masses.dtype, float))
        incoming_names = [spec.name for spec in particle.species]
        if masses.ndim != 1 or len(masses) != len(incoming_names):
            raise ValueError("Particle masses must be a 1D row matching species")
        if len(set(incoming_names)) != len(incoming_names):
            raise ValueError("Particle species names must be unique")
        if len(self.ids) == 0:
            return masses.copy()
        names = [spec.name for spec in self.species]
        if len(set(names)) != len(names) or set(names) != set(incoming_names):
            raise ValueError("Incoming particle must have the population species set")
        indices = {name: i for i, name in enumerate(incoming_names)}
        missing = object()
        for spec in self.species:
            incoming = particle.species[indices[spec.name]]
            if spec is incoming:
                continue
            # Preserve name-only species objects; compare physical fields when present.
            for field in ("density", "kappa", "molar_mass", "surface_tension"):
                left = getattr(spec, field, missing)
                right = getattr(incoming, field, missing)
                if not _same_field_value(left, right):
                    raise ValueError(f"Incompatible {field} for species {spec.name}")
        return masses[[indices[name] for name in names]].copy()

    def set_particle(self, particle, part_id, num_conc, suppress_warning=True):
        part_id = int(part_id)
        if part_id not in self.ids:
            if not suppress_warning:
                warn('part_id not in self.ids, adding ' + str(part_id))
            self.add_particle(particle, part_id, num_conc)
        else:
            masses = self._aligned_particle_masses(particle)
            concentration = np.asarray(num_conc)
            if concentration.ndim == 0 and np.issubdtype(concentration.dtype, np.floating):
                num_conc = concentration[()]
            else:
                num_conc = float(num_conc)
            idx = self.find_particle(part_id)
            # Integer-backed inputs must not silently truncate fractional updates.
            self.spec_masses = np.asarray(
                self.spec_masses, dtype=np.result_type(self.spec_masses.dtype, masses.dtype, float)
            )
            self.num_concs = np.asarray(
                self.num_concs, dtype=np.result_type(self.num_concs.dtype, np.asarray(num_conc).dtype, float)
            )
            self.spec_masses[idx,:] = masses
            self.num_concs[idx] = num_conc

    def add_particle(self, particle, part_id, num_conc):
        masses = self._aligned_particle_masses(particle)
        if len(self.ids) == 0:
            self.species = particle.species
            self.spec_masses = np.zeros([1,len(particle.species)], dtype=masses.dtype)
            self.spec_masses[0,:] = masses
            self.num_concs = np.hstack([num_conc])
            self.ids = [part_id]
        else:
            # ids may be a tuple; build the new state before committing it.
            spec_masses = np.vstack([self.spec_masses, masses.reshape(1,-1)])
            num_concs = np.hstack([self.num_concs, num_conc])
            ids = list(self.ids) + [part_id]
            self.spec_masses, self.num_concs, self.ids = spec_masses, num_concs, ids
    
    def get_species_idx(self, spec_name):
        names = [spec.name for spec in self.species]
        try:
            return np.where(np.array(names) == spec_name)[0][0]
        except:
            return None
    
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
    
    def reduce_mixing_state(self, mixing_state='part_res', 
                            RH=None, T=None, 
                            sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
        raise NotImplementedError(
            "ParticlePopulation.reduce_mixing_state is intentionally unsupported in "
            "part2pop 1.0.0. The legacy implementation was unreliable and could raise "
            "incidental TypeError/AxisError/AttributeError. Mixing-state reduction "
            "behavior is deferred to a future release after scientific specification "
            "and validation are completed."
        )
            
    def clone_detached(self):
        """Return a copy that shares immutable data but has detached numeric arrays."""
        return ParticlePopulation(
            species=self.species, spec_masses=self.spec_masses.copy(),
            num_concs=self.num_concs.copy(), ids=self.ids, classes=self.classes, species_modifications=self.species_modifications
        )
