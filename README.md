# part2pop

[![CI](https://github.com/pnnl/part2pop/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pnnl/part2pop/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pnnl/part2pop/branch/main/graph/badge.svg)](https://codecov.io/gh/pnnl/part2pop)

A unified analysis framework for standardizing aerosol populations from models, observations, and parameterizations.

`part2pop` is a lightweight Python library that provides a standardized representation of aerosol particles and populations. By translating model-derived, observation-constrained, and parameterized population descriptions into a common particle-population interface, particle-level process models and diagnostics can be applied consistently across data sources. Its modular, registry-based architecture keeps extension points explicit, supporting extensible population builders, reusable analyses, and maintainable development by both researchers and AI coding assistants.

## Features

- **Standardized aerosol representation** via `AerosolSpecies`, `Particle`, and `ParticlePopulation`
- **Species registry** with density, hygroscopicity (`kappa`), molar mass, and surface tension
- **Population builders** for monodisperse, lognormal, sampled, observation-based, and model-derived populations
- **Optical-property builders** supporting homogeneous spheres, core-shell particles, and fractal aggregates
- **Freezing-property builders** for immersion-freezing metrics
- **Analysis utilities** for size distributions, hygroscopic growth, CCN activation, optical properties, freezing diagnostics, and bulk moments
- **Visualization tools** for time series, size distributions, optical coefficients, and freezing curves

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

## Documentation

User documentation is available in [`docs/`](https://github.com/pnnl/part2pop/tree/main/docs), including:

- [Population builders](https://github.com/pnnl/part2pop/blob/main/docs/population_builders.md)
- [Analysis variables](https://github.com/pnnl/part2pop/blob/main/docs/analysis_variables.md)
- [Visualization](https://github.com/pnnl/part2pop/blob/main/docs/visualization.md)
- [Optics](https://github.com/pnnl/part2pop/blob/main/docs/optics.md)
- [Freezing](https://github.com/pnnl/part2pop/blob/main/docs/freezing.md)

## Contributing and support

Please use GitHub Issues for bug reports, questions, and feature requests. See [CONTRIBUTING.md](https://github.com/pnnl/part2pop/blob/main/CONTRIBUTING.md) for contribution guidelines.

## Research use and integrations

`part2pop` is used as a shared aerosol-population layer in related research software.

- In [LD-Chem](https://github.com/pnnl/LD-Chem), `part2pop` is a core dependency for constructing particle populations used as model input and for analyzing model output.
- In [AMBRS](https://github.com/AMBRS-project/ambrs), `part2pop` provides a unified population representation and diagnostic framework for comparing aerosol box-model output.

## License

`part2pop` is distributed under the MIT License. See the [LICENSE](https://github.com/pnnl/part2pop/blob/main/LICENSE) file for details.
