# part2pop

{{CODECOV_BADGE}}

> A modular Python toolkit for building, analyzing, and extending aerosol particle populations.

`part2pop` is a lightweight Python library that provides a **standardized representation of aerosol particles and populations**, together with modular builders for species, particle populations, optical properties, freezing properties, and analysis tools. Its **builder/registry design** makes the system easily extensible: new population types, particle morphologies, freezing parameterizations, or species definitions can be added by placing small modules into the appropriate `factory/` directory—without modifying core code.

The framework enables reproducible process-level investigations, sensitivity studies, and intercomparison analyses across diverse model and observational datasets by providing a consistent interface for particle-resolved, model-derived, and parameterized aerosol populations.

---

## Features

### 🔹 Standardized aerosol representation
- `AerosolSpecies`, `AerosolParticle`, and `ParticlePopulation` provide a unified way to store particle composition, size, and per-particle number concentration.
- Particle properties (wet/dry diameters, hygroscopicity, mass fractions, kappa) are computed using consistent physics.

### 🔹 Species registry
- Built-in species definitions with density, refractive index, κ-Köhler hygroscopicity, molar mass, and surface tension.
- User overrides (e.g., modify kappa or refractive index) allowed at load time.

### 🔹 Population builders
Modular builders convert configuration dictionaries to particle populations:
- `monodisperse`
- `binned_lognormals`
- `sampled_lognormals`
- `partmc` loader
- `mam4` loader

All produce a `ParticlePopulation` with:
- per-particle species masses,
- number concentrations,
- consistent particle IDs.

### 🔹 Optical-property builders
Morphology-specific routines compute scattering, absorption, extinction, and asymmetry parameter (`g`) across wavelength and RH grids.

Supported morphologies include:
- Homogeneous spheres  
- Core–shell particles  
- Fractal aggregates  

Optical builders can call external packages such as **PyMieScatt** or **pyBCabs** when available.

### 🔹 Freezing-property builders
- Immersion-freezing metrics for individual particles and populations.
- Configurable parameterizations via `freezing/factory/`.

### 🔹 Analysis utilities
Convenience functions for:
- size distributions (`dN/dlnD`)
- hygroscopic growth
- CCN activation spectra
- particle- and population-level moments
- bulk composition and mass fractions

### 🔹 Visualization tools
Plotting helpers for size distributions, optical coefficients, freezing curves, and more.

---

## Installation

```bash
git clone https://github.com/pnnl/part2pop.git
cd part2pop
pip install -e .
```

Optional dependencies (e.g., `netCDF4`, `PyMieScatt`, `pyBCabs`) enable extended IO and optical capabilities.  
Missing optional dependencies generate clear, informative error messages.

---

## Quick start

### Build a simple population

```python
from part2pop.population.builder import build_population

config = {
    "type": "monodisperse",
    "diameter": 0.2e-6,
    "species_masses": {"SO4": 1e-15, "BC": 5e-16},
}

pop = build_population(config)
print(pop)
```

### Compute optical properties

```python
from part2pop.optics import build_optical_population

opt_pop = build_optical_population(pop, {
    "type": "homogeneous",
    "wvl_grid": [550e-9],
    "rh_grid": [0.0],
})

print(opt_pop.get_optical_coeff("b_scat", rh=0.0, wvl=550e-9))
```

### Analyze a population

```python
from part2pop.analysis import size_distribution

d, dNdlnD = size_distribution(pop)
```

More examples are available under `examples/`.

---

## Repository structure

```
src/part2pop/
    aerosol_particle.py      # Particle representation and physics
    species/                 # Species registry + built-in datasets
    population/              # Population builders + factories
    optics/                  # Optical property builders and morphologies
    freezing/                # Immersion-freezing parameterizations
    analysis/                # Derived quantities and utilities
    viz/                     # Plotting helpers
    data/                    # Packaged species data
```

---

## Contributing

The design philosophy of `part2pop` is that **all extensibility happens through factories**.

To add new functionality:

- **New species definition:**  
  Add a module to `species/` or `species/factory/`, then register it.

- **New population type:**  
  Add a module under `population/factory/` with a `build(config)` function.

- **New optical morphology:**  
  Add a module to `optics/factory/` and register it.

- **New freezing parameterization:**  
  Add a module under `freezing/factory/`.

No changes to the core API are required.  
Please open an issue or PR to discuss proposed additions.

---

## License

See the `LICENSE` file in this repository.

