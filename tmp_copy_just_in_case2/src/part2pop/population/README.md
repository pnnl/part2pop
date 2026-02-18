# part2pop.population

Purpose: Population dataclass and builders for constructing particle populations from various inputs.

Contents

- `base.py` — `ParticlePopulation` dataclass (species, masses, concentrations, ids) with utility methods
- `builder.py` — `build_population(config)` using factory discovery
- `factory/` — registry and concrete builders:
  - `registry.py` — `register()` decorator and `discover_population_types()`
  - `monodisperse.py` — monodisperse population builder
  - `binned_lognormals.py` — multi-mode binned lognormal builder
  - `partmc.py` — PARTMC NetCDF reader -> population
  - `mam4.py` — MAM4 output -> binned lognormal population

Notes

- Builders accept a `config` dict; common keys include species names/fractions and distribution parameters.
