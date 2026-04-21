# part2pop.species

Purpose: Define aerosol species properties and provide a registry/lookup mechanism.

Contents

- `base.py` — `AerosolSpecies` dataclass; loads defaults from `datasets/species_data/aero_data.dat` when needed
- `registry.py` — `AerosolSpeciesRegistry` singleton; helpers `get_species`, `register_species`, `extend_species`, `list_species`, `retrieve_one_species`

Data

- Expects `datasets/species_data/aero_data.dat` present in the repository (or accessible via `data_path`).
