# part2pop

A unified analysis framework for standardizing aerosol populations from models, observations, and parameterizations.

`part2pop` is a lightweight Python library that provides a standardized representation of aerosol particles and populations. By translating model-derived, observation-constrained, and parameterized population descriptions into a common particle-population interface, `part2pop` lets particle-level process models and diagnostics be applied consistently across data sources. Its modular, registry-based architecture supports extensible population builders, reusable analyses, and AI-assisted development workflows.

The framework supports reproducible process-level investigations, sensitivity studies, and intercomparison analyses by providing a consistent interface for aerosol populations derived from models, measurements, and parameterized distributions.

## Features

- **Standardized aerosol representation** via `AerosolSpecies`, `Particle`, and `ParticlePopulation`
- **Species registry** with density, hygroscopicity (`kappa`), molar mass, and surface tension
- **Population builders** for monodisperse, lognormal, sampled, observation-based, and model-derived populations
- **Optical-property builders** supporting homogeneous spheres, core-shell particles, and fractal aggregates
- **Freezing-property builders** for immersion-freezing metrics
- **Analysis utilities** for size distributions, hygroscopic growth, CCN activation, optical properties, and bulk moments
- **Visualization tools** for size distributions, optical coefficients, and freezing curves

Optional external packages, including `PyMieScatt`, `pyBCabs`, and `netCDF4`, are used when available.

## Installation

```bash
pip install part2pop
````

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

For end-to-end analysis and plotting workflows, see the notebooks in the [examples directory](https://github.com/pnnl/part2pop/tree/main/examples).

## Interactive viewer

An experimental Streamlit UI lets you build populations through the factory registries and render existing visualization builders.

For a source checkout, run:

```bash
pip install -e .
streamlit run scripts/launch_viewer.py
```

The viewer source lives under `viewer/`. The sidebar lists registered population and plot types; choose a population and visualization workflow, adjust the metadata-driven controls, and render figures and diagnostics interactively.

## Contributing

`part2pop` is designed so that new population types, optical morphologies, freezing parameterizations, diagnostics, and visualization types can be added through factory/registry modules without changing the core population representation.

Please open an issue or pull request to discuss proposed additions.

See the [contribution guidelines](https://github.com/pnnl/part2pop/blob/main/CONTRIBUTING.md) for contribution and support information.

## License

See the [LICENSE](https://github.com/pnnl/part2pop/blob/main/LICENSE) file in the repository.
