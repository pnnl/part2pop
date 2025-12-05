# pyparticle

[![CI](https://github.com/pnnl/pyparticle/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pnnl/pyparticle/actions/workflows/ci.yml)

A Python library for constructing aerosol particle populations, attaching species-level physical properties, building per-particle morphologies, and aggregating to population-level aerosol properties. The package uses factory/builder discovery, so new population types, aerosol species, and morphologies can be added by dropping small modules into `factory/` folders.

> **Note:** The distribution and import name are both **`pyparticle`** (PEP 8). If you previously imported `PyParticle`, switch to `pyparticle`.

# Install

## Create a dev environment

```
conda env create -f environment.yml -n pyparticle
conda activate pyparticle
```

## Editable install

```
pip install -e .
```

## Optional extras (used by some examples/tests)

* PyMieScatt (used for optics calculations)
* netCDF4 (used to construct populations from aerosol model output)

Install them in the same environment if you need those features:

```
pip install PyMieScatt netCDF4
```

# Quickstart

Build a population → attach morphology (if needed) → query an aerosol property.

Example (optics shown here; the same pattern applies to freezing):

```python
from PyParticle.population.builder import build_population
from PyParticle.optics.builder import build_optical_population

# 1) Build a simple binned lognormal population (single species: SO4)
pop_cfg = {
    "type": "binned_lognormals",
    "GMD": [100e-9],               # meters
    "GSD": [1.6],
    "N":   [1e8],                  # m^-3
    "aero_spec_names": [["SO4"]],
    "aero_spec_fracs": [[1.0]],
    "N_bins": 60,
    "species_modifications": {"SO4": {"density": 1770, "n_550": 1.45, "k_550": 0.0}}
}
pop = build_population(pop_cfg)

# 2) Build optics (homogeneous spheres) on an RH/λ grid
opt_cfg = {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]}
opt_pop = build_optical_population(pop, opt_cfg)

# 3) Query scattering coefficient at RH=0, λ=550 nm (SI units inside: meters)
b_scat = opt_pop.get_optical_coeff("b_scat", rh=0.0)  # numpy array or float depending on wvl_grid defined in opt_cfg
print(b_scat)
```

# Concepts & Architecture (brief)

* **Particle**: lightweight object holding species and per-species masses; exposes helpers like dry/wet diameters and κ.
* **ParticlePopulation**: container for many `Particle` items with number concentrations and IDs; carries `species_modifications: Dict[str, dict]` for runtime overrides.
* **Derived properties**:

  * **OpticalParticle / OpticalPopulation**: wraps base particles to compute per-particle optical cross-sections (Csca, Cabs, Cext, g) and aggregates to optical coefficients (b_scat, b_abs, etc.).
  * **CCN (cloud condensation nuclei)**: water uptake and activation are computed on the **base** `Particle` / `ParticlePopulation`.
  * **FreezingParticle / FreezingPopulation**: wraps base particles to evaluate heterogeneous ice nucleation properties (e.g., Jhet/IN metrics) and aligns results with population IDs.

# Discovery / extension points

* **Population types**: add a module under `src/PyParticle/population/factory/` exposing a `build(config)` callable. The population builder auto-discovers modules in that folder.
* **Species default**: define default species in `src/PyParticle/species/factory.py`.
* **Optics morphologies**: add a module under `src/PyParticle/optics/factory/` and register a build callable (the registry or module-level `build` will be discovered).
* **Freezing morphologies**: add a module under `src/PyParticle/freezing/factory/` and register a build callable (the registry or module-level `build` will be discovered).

Developer guidance and templates are available in `docs/developer/factories.md`.

# Repository layout (high level)

* `src/PyParticle/` — core library

  * `aerosol_particle.py`
  * `population/` (builder, base, factories)
  * `optics/` (builder, base, refractive_index, factories)
  * `species/` (registry and data readers)
  * `freezing/` (builder, base, factories)
  * `analysis/` (particle- and population-level)
  * `viz/` (plotting helpers)
* `examples/`
* `datasets/` (species_data/, model_data/)
* `tests/`
* `docs/`
* `environment.yml`, `setup.py`, `pyproject.toml`, `tools/`

# Testing

Run unit tests locally with the conda env active:

```
pytest -q
```

# Contributing

1. Fork or branch from `main`/`develop`.
2. Add tests (unit tests for new behavior; integration tests when optional deps apply).
3. Run the test suite locally and ensure examples still run.
4. Submit a PR with a clear description and rationale.

# License

See `LICENSE` in the repository root.

# Acknowledgments

The PyParticle architecture was developed under the Integrated Cloud, Land-surface, and Aerosol System Study (ICLASS) project with support from the U.S. Department of Energy's Atmospheric System Research. Development and optics work were supported in part by Pacific Northwest National Laboratory.
