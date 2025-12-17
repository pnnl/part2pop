# part2pop

A modular Python toolkit for scaling particle-level models to predict population-level aerosol effects.

`part2pop` is a lightweight Python library that provides a **standardized representation of aerosol particles and populations**, together with modular builders for species, particle populations, optical properties, freezing properties, and analysis tools.

Its **builder/registry design** makes the system easily extensible: new population types, particle morphologies, freezing parameterizations, or species definitions can be added by placing small modules into the appropriate factory directory—without modifying core code.

The framework enables reproducible process-level investigations, sensitivity studies, and intercomparison analyses across diverse model and observational datasets by providing a consistent interface for aerosol populations derived from models, measurements, or parameterized distributions.

## Features

- **Standardized aerosol representation** via `AerosolSpecies`, `AerosolParticle`, and `ParticlePopulation`
- **Species registry** with density, refractive index, hygroscopicity, and thermodynamic properties
- **Population builders** for monodisperse, lognormal, sampled, and model-derived populations
- **Optical-property builders** supporting homogeneous spheres, core–shell particles, and fractal aggregates
- **Freezing-property builders** for immersion-freezing metrics
- **Analysis utilities** for size distributions, hygroscopic growth, CCN activation, and bulk moments
- **Visualization tools** for size distributions, optical coefficients, and freezing curves

Optional external packages (e.g., PyMieScatt, pyBCabs, netCDF4) are used when available.

## Installation

```bash
pip install part2pop
```

For development:

```bash
git clone https://github.com/pnnl/part2pop.git
cd part2pop
pip install -e .
```

## Quick start

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

More examples are available in the project repository.

## Contributing

`part2pop` is designed so that **all extensibility happens through factories**.

New population types, optical morphologies, freezing parameterizations, diagnostics, and visualization types can be added by placing small modules in the appropriate factory directories without changing the core API.

Please open an issue or pull request to discuss proposed additions.

## License

See the `LICENSE` file in the repository.
