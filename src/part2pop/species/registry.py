"""Runtime AerosolSpecies registry and file-based fallback lookup.

This module provides an in-memory registry allowing users to register
custom species at runtime and a `retrieve_one_species` fallback that
reads `datasets/species_data/aero_data.dat` for default species.
"""

import copy
from .base import AerosolSpecies
from .resolution import resolve_species_name
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

    def get(self, name: str, specdata_path: str | None, **modifications) -> AerosolSpecies:
        """Get a species from the registry, optionally with modifications.
        Falls back to data file lookup if not registered.
        """
        key = str(name).upper()
        if key in self._custom:
            base = copy.deepcopy(self._custom[key])
            for k, v in modifications.items():
                setattr(base, k, v)
            return base

        resolved_name = resolve_species_name(name)
        resolved_key = resolved_name.upper()
        if resolved_key in self._custom:
            base = copy.deepcopy(self._custom[resolved_key])
            for k, v in modifications.items():
                setattr(base, k, v)
            return base
        
        # fallback to retrieve_one_species (file-based) if not registered
        try:
            return retrieve_one_species(
                resolved_name,
                specdata_path=specdata_path,
                spec_modifications=modifications,
            )
        except ValueError as exc:
            if str(name) != resolved_name:
                raise ValueError(
                    f"Species lookup failed for {name!r} (resolved to {resolved_name!r}): {exc}"
                ) from exc
            raise

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

def get_species(name: str, specdata_path: str | None, **modifications) -> AerosolSpecies:
    return _registry.get(name, specdata_path, **modifications)

def list_species():
    return _registry.list_species()


def describe_species(name: str, specdata_path: str | None = None):
    key = str(name).upper()
    if key in _registry._custom:
        sp = _registry._custom[key]
        return {
            "name": sp.name,
            "module": sp.__class__.__module__,
            "type": sp.__class__.__name__,
            "description": (sp.__class__.__doc__ or "").strip() or None,
            "defaults": {
                "density": getattr(sp, "density", None),
                "kappa": getattr(sp, "kappa", None),
                "molar_mass": getattr(sp, "molar_mass", None),
                "surface_tension": getattr(sp, "surface_tension", None),
            },
        }

    resolved_name = resolve_species_name(name)
    resolved_key = resolved_name.upper()
    if resolved_key in _registry._custom:
        sp = _registry._custom[resolved_key]
        return {
            "name": sp.name,
            "module": sp.__class__.__module__,
            "type": sp.__class__.__name__,
            "description": (sp.__class__.__doc__ or "").strip() or None,
            "defaults": {
                "density": getattr(sp, "density", None),
                "kappa": getattr(sp, "kappa", None),
                "molar_mass": getattr(sp, "molar_mass", None),
                "surface_tension": getattr(sp, "surface_tension", None),
            },
        }

    try:
        sp = retrieve_one_species(
            resolved_name,
            specdata_path=specdata_path,
            spec_modifications={},
        )
    except ValueError as exc:
        available = ", ".join(sorted(list_species())) or "<none>"
        raise ValueError(
            f"Unknown species: {name}. Custom-registered species: {available}"
        ) from exc

    return {
        "name": sp.name,
        "module": sp.__class__.__module__,
        "type": sp.__class__.__name__,
        "description": (sp.__class__.__doc__ or "").strip() or None,
        "defaults": {
            "density": getattr(sp, "density", None),
            "kappa": getattr(sp, "kappa", None),
            "molar_mass": getattr(sp, "molar_mass", None),
            "surface_tension": getattr(sp, "surface_tension", None),
        },
    }

def extend_species(species: AerosolSpecies):
    _registry.extend(species)

def _iter_aero_data_lines(specdata_path=None):
    
    if specdata_path is None:
        with open_dataset('species_data/aero_data.dat') as fh:
            # "species_data/aero_data.dat") as fh:
            for line in fh:
                yield line
    else:
        with open_dataset(specdata_path+'/aero_data.dat') as fh:
            # "species_data/aero_data.dat") as fh:
            for line in fh:
                yield line
    

def retrieve_one_species(name, specdata_path=None, spec_modifications={}):
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
    
    for line in _iter_aero_data_lines(specdata_path=specdata_path):
        
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
