"""Builder helpers to create optical particles and optical populations.

This module wraps a morphology discovery registry to construct per-particle
optical objects and aggregate them into an `OpticalPopulation`.
"""

from .factory.registry import discover_morphology_types
from .base import OpticalPopulation
# from .. import data_path


class OpticalParticleBuilder:
    """Construct an optical particle instance from a config.

    Parameters
    ----------
    config : dict
        Configuration dictionary with a 'type' key indicating morphology.
    """
    def __init__(self, config):
        self.config = config
    
    def build(self, base_particle):
        type_name = self.config.get("type")
        if not type_name:
            raise ValueError("Config must include a 'type' key.")
        types = discover_morphology_types()
        if type_name not in types:
            raise ValueError(f"Unknown optics morphology type: {type_name}")
        cls_or_factory = types[type_name]
        # Expect a class or callable that accepts (base_particle, config)
        return cls_or_factory(base_particle, self.config)


def build_optical_particle(base_particle, config):
    """Helper: build and return an optical particle from base particle and config."""
    return OpticalParticleBuilder(config).build(base_particle)


def build_optical_population(base_population, config):
    """Build an OpticalPopulation from a base ParticlePopulation and config.

    Parameters
    ----------
    base_population : ParticlePopulation
        Base population containing species, masses, concentrations, and ids.
    config : dict
        Optics configuration (rh_grid, wvl_grid, and morphology type).

    Returns
    -------
    OpticalPopulation
        Aggregated optics for the population.
    """
    rh_grid = config.get('rh_grid', [0.0])
    wvl_grid = config.get('wvl_grid', [550e-9])
    
    # Attach wavelength-aware refractive indices once for each species in the
    # base population using the optics wavelength grid and any provided
    # species_modifications. Doing this here avoids rebuilding RIs per-particle
    # and centralizes the decision point where the wavelength grid is known.
    # Prefer species_modifications from optics config; if absent, fall back to
    # the base population's recorded modifications (set by population builders).
    species_mods = config.get('species_modifications')
    if not species_mods:
        species_mods = getattr(base_population, 'species_modifications', {}) or {}
    if species_mods is None:
        species_mods = {}
    # (no debug printing)
    # specdata_path = config.get('specdata_path', None) or data_path / 'species_data'
    # specdata_path = '../species/species_data'
    # Import here to avoid circular imports at module import time
    from .refractive_index import build_refractive_index

    for spec in base_population.species:
        # Allow SOA envelope fallback (mirror logic used elsewhere)
        soa_names = {'MSA','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2'}
        soa_mods = species_mods.get('SOA', {})
        mods = species_mods.get(spec.name, {})
        if not mods and spec.name in soa_names:
            mods = soa_mods
        # no debug output
        # Only attach if species doesn't already have a wavelength-aware RI
        existing = getattr(spec, 'refractive_index', None)
        try:
            # If existing has functions for the requested grid we skip rebuild.
            skip = False
            if existing is not None and hasattr(existing, 'real_ri_fun') and hasattr(existing, 'imag_ri_fun'):
                skip = True
        except Exception:
            skip = False
        if not skip:
            build_refractive_index(spec, wvl_grid, modifications=mods)

    # Pass the base population so OpticalPopulation can inherit ids/num_concs/etc.
    optical_population = OpticalPopulation(base_population, rh_grid, wvl_grid)

    for part_id in base_population.ids:
        base_particle = base_population.get_particle(part_id)
        optical_particle = build_optical_particle(base_particle, config)
        optical_population.add_optical_particle(optical_particle, part_id)

    return optical_population
