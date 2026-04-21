# part2pop.optics

Purpose: Optical particle abstractions, builders, utilities, and morphology implementations.

Contents

- `base.py` — `OpticalParticle` (ABC) and `OpticalPopulation` (array-backed, aggregated coefficients)
- `builder.py` — `build_optical_particle`, `build_optical_population` via factory discovery
- `utils.py` — mapping of optics type names, array extraction helpers
- `refractive_index.py` — simple refractive index class and stub function
- `population.py` — legacy/alternate `OpticalPopulation` (list of particles, aggregator)
- `factory/` — morphology registry and implementations:
  - `registry.py` — `register()` decorator and `discover_morphology_types()`
  - `core_shell.py` — core-shell morphology implementation
  - `homogeneous.py` — homogeneous sphere morphology implementation
  - `fractal.py` — placeholder fractal aggregate

Usage

- Provide a base population and a config dict with keys like `type`, `rh_grid`, `wvl_grid` (meters), and options specific to each morphology.
- Example: see `examples/*_binned_lognormal.py`.
