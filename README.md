# part2pop

[![CI](https://github.com/pnnl/part2pop/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pnnl/part2pop/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pnnl/part2pop/branch/main/graph/badge.svg)](https://codecov.io/gh/pnnl/part2pop)

A unified analysis framework for standardizing aerosol populations from models, observations, and parameterizations.

`part2pop` is a lightweight Python library that provides a standardized representation of aerosol particles and populations. By translating model-derived, observation-constrained, and parameterized population descriptions into a common particle-population interface, `part2pop` lets particle-level process models and diagnostics be applied consistently across data sources. Its modular, registry-based architecture supports extensible population builders, reusable analyses, and AI-assisted development workflows.

## Features

- **Standardized aerosol representation** via `AerosolSpecies`, `Particle`, and `ParticlePopulation`
- **Species registry** with density, hygroscopicity (`kappa`), molar mass, and surface tension
- **Population builders** for monodisperse, lognormal, sampled, observation-based, and model-derived populations
- **Optical-property builders** supporting homogeneous spheres, core-shell particles, and fractal aggregates
- **Freezing-property builders** for immersion-freezing metrics
- **Analysis utilities** for size distributions, hygroscopic growth, CCN activation, and bulk moments
- **Visualization tools** for size distributions, optical coefficients, and freezing curves

Optional external packages (e.g., PyMieScatt, pyBCabs, netCDF4) are used when available.

## Installation

```bash
pip install part2pop
```

## Quick start

```python
from part2pop.population.builder import build_population

config = {
    "type": "monodisperse",
    "N": [1.0e8],
    "D": [0.2e-6],
    "aero_spec_names": [["SO4", "BC"]],
    "aero_spec_fracs": [[0.7, 0.3]],
    "D_is_wet": False,
}

pop = build_population(config)

print(pop.get_Ntot())
print([species.name for species in pop.species])
```

For end-to-end analysis and plotting workflows, see the notebooks in `examples/`.

See the [contribution guidelines](https://github.com/pnnl/part2pop/blob/main/CONTRIBUTING.md) for contribution and support information.
