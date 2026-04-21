# part2pop package

Purpose: Core package that exposes particle, species, population, and optics builders.

Contents

- `__init__.py` — package exports (Particle, make_particle, species registry, build_population, optics builders)
- `aerosol_particle.py` — `Particle` dataclass and helper functions (wet/dry diameter, kappa, masses)
- `utilities.py` — parsing and lognormal moment utilities
- `storer.py` — serialization helpers for optical outputs and species dictionaries
- `constants.py` — physical constants
- Subpackages:
  - `optics/` — optical particle base/population, builders, and morphologies
  - `population/` — population base and builders (monodisperse, binned_lognormals, partmc, mam4)
  - `species/` — species dataclass and registry

Exports (selected)

- `Particle`, `make_particle`, `make_particle_from_masses`
- `AerosolSpecies`, `get_species`, `register_species`, `list_species`, `extend_species`
- `ParticlePopulation`, `build_population`
- `build_optical_particle`, `build_optical_population`
