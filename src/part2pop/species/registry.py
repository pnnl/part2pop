"""Runtime AerosolSpecies registry and file-based fallback lookup.

This module provides an in-memory registry allowing users to register
custom species at runtime and a `retrieve_one_species` fallback that
reads `datasets/species_data/aero_data.dat` for default species.
"""

import copy
from .base import AerosolSpecies
# from ..data import species_open
from ..data import open_dataset
import os


class AerosolSpeciesRegistry:
    def __init__(self):
        # Maps uppercase name to AerosolSpecies
        self._custom = {}

    def register(self, species: AerosolSpecies):
        """Add or update a species in the registry."""
        self._custom[species.name.upper()] = copy.deepcopy(species)

    def get(self, name: str, **modifications) -> AerosolSpecies:
        """Get a species from the registry, optionally with modifications.
        Falls back to data file lookup if not registered.
        """
        key = name.upper()
        if key in self._custom:
            base = copy.deepcopy(self._custom[key])
            for k, v in modifications.items():
                setattr(base, k, v)
            return base
        
        # fallback to retrieve_one_species (file-based) if not registered
        return retrieve_one_species(name, spec_modifications=modifications)

    def extend(self, species: AerosolSpecies):
        """Alias for register for API clarity."""
        self.register(species)

    def list_species(self):
        """List only custom-registered species."""
        return list(self._custom.keys())

# Singleton instance for package-wide use
_registry = AerosolSpeciesRegistry()

def register_species(species: AerosolSpecies):
    _registry.register(species)

def get_species(name: str, **modifications) -> AerosolSpecies:
    return _registry.get(name, **modifications)

def list_species():
    return _registry.list_species()

def extend_species(species: AerosolSpecies):
    _registry.extend(species)

def _iter_aero_data_lines(): 
    
    with open_dataset('species_data/aero_data.dat') as fh:
        # "species_data/aero_data.dat") as fh:
        for line in fh:
            yield line

def retrieve_one_species(name, spec_modifications={}):
    """Retrieve a species from data file and apply optional modifications.

    Parameters
    ----------
    name : str
        Species name to lookup (case-insensitive).
    spec_modifications : dict
        Optional overrides for species properties (kappa, density, etc.).

    Returns
    -------
    AerosolSpecies
        Constructed species dataclass.
    """
    for line in _iter_aero_data_lines():
        
        if line.strip().startswith("#"):
            continue
        if line.upper().startswith(name.upper()):
            parts = line.split()
            if len(parts) < 5:
                continue
            name_in_file, density, ions_in_solution, molar_mass, kappa = parts[:5]
            
            # Apply modifications if provided
            kappa = spec_modifications.get('kappa', kappa)
            density = spec_modifications.get('density', density)
            surface_tension = spec_modifications.get('surface_tension', 0.072)
            molar_mass_val = spec_modifications.get('molar_mass', molar_mass)

            return AerosolSpecies(
                name=name,
                density=float(density),
                kappa=float(kappa),
                molar_mass=float(str(molar_mass_val).replace('d','e')),
                surface_tension=float(surface_tension)
            )

    raise ValueError(f"Species data for '{name}' not found in data file.")
