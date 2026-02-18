---
title: "part2pop: modular tool to build and analyze aerosol particle populations"
author:
  - "Laura Fierce"
  - "Payton Beeler"
tags:
  - Python
  - aerosols
  - atmospheric science
  - aerosol-cloud interactions
  - aerosol-radiation interactions
authors:
  - name: Laura Fierce
    orcid: "0000-0002-8682-1453"
    affiliation: 1
  - name: Payton Beeler
    orcid: "0000-0003-4759-1461"
    affiliation: 1
affiliations:
  - name: Pacific Northwest National Laboratory
    index: 1
bibliography: paper.bib
link-citations: true
nocite: |
  @*
---

# Summary

**part2pop** is a lightweight Python library for describing and analyzing aerosol particle populations. The package includes modular builders for aerosol species, particle populations, and particle morphologies, which interface with existing models for derived aerosol properties, such as cloud condensation nuclei (CCN) activity, ice-nucleating particle (INP) potential, and optical properties. Its **builder/registry** design allows new aerosol species, population types, and morphologies to be added by adding small modules to the appropriate `factory/` folders without modifying code API. This modular, reproducible framework facilitates process-level investigations, sensitivity analyses, and intercomparison studies across diverse model and observational datasets. 

The core components include:

- **AerosolSpecies**, **AerosolParticle**, **ParticlePopulation** classes that provide a standardized representation of aerosol particles from diverse data sources.

- **Species builder** that supplies physical properties (e.g., density, hygroscopicity parameter, refractive index) for aerosol species, with optional per-species overrides.

- **Population builders** for parametric and model-derived populations (e.g., *binned lognormal*, *monodisperse*, and loaders for *MAM4* and *PartMC* simulation output).

- **Optical particle builders** that compute wavelength- and RH-dependent optical properties for different particle morphologies (e.g., *homogeneous*, *core–shell*, *fractal*) using existing libraries

- **Freezing particle builders** that estimate INP-relevant metrics from particle composition and size.

- An **analysis module** that calculates particle- and population-level variables from part2pop populations.

- A **viz** package to generate figures from part2pop populations.

Example scripts demonstrate (i) optical properties for lognormal mixtures, (ii) optical properties for lognormal mixtures with different morphologies (iii) comparisons of CCN activity between MAM4- and PartMC-derived populations, and (iv) freezing-oriented calculations on common temperature/RH grids.

# Statement of need
The physical properties of aerosols must be well quantified for a variety of atmospheric, air quality, and industrial applications. A wide range of tools have been developed to simulate [e.g., @Riemer2009_JGR_PartMC; @Liu2016_GMD_MAM4; @Bauer2008_ACP_MATRIX; @Zaveri2008_JGR_MOSAIC] and observe [e.g., @Jayne2000_AST_AMS; @DeCarlo2006_AnalChem_HRToFAMS; @Knutson1975_JAS_DMA; @Wang1990_AST_SEMS; @Schwarz2006_JGR_SP2; @Schwarz2008_JGR_SP2; @Zelenyuk2015_JASMS_miniSPLAT] aerosol particle populations, producing varied aerosol data that is often difficult to compare directly. **part2pop** provides a standardized description of aerosol particle populations and facilitates evaluation of derived aerosol properties.

**Leveraging existing models.** part2pop is designed to interoperate with established aerosol-property models rather than reimplementing them. The current implementation focuses on aerosol effects relevant for the atmosphere, with many functions adapted from previous studies of CCN activity [@Fierce2013_JGR; @Fierce2017_JGR; @Fierce2024_JAS], optical properties [@Fierce2016_NatComm; @Fierce2020_PNAS], mixing state approximations [@Fierce2015_ACP; @Fierce2017_BAMS_mixing_state;@Fierce2024_JAS], and heterogeneous ice nucleation [@AlpertKnopf2016_ACP]. For optical calculations, part2pop can call external packages such as PyMieScatt [@PyMieScatt] and pyBCabs [@beeler2022acp]. For hygroscopicity and CCN-relevant calculations it follows the kappa-Köhler framework [@Petters2007_ACP], treating kappa as a per-species property that can be supplied by the species registry or overridden at runtime. Immersion freezing is represented using the stochastic immersion freezing model [@KnopfAlpert2013_Faraday;@AlpertKnopf2016_ACP]. Model loaders (e.g., MAM4, PartMC) convert model outputs to the part2pop internal representation so downstream analyses (CCN, optics, freezing) can use the same utilities. Where third-party packages are optional (e.g., `netCDF4` for NetCDF I/O or PyMieScatt for optics calculations), part2pop raises explicit errors with clear remediation so analyses are deterministic and reproducible.

**Modular structure.** The codebase follows a strict builder/registry pattern so new capabilities are added by dropping a single module into a `factory/` folder. Modules for particle populations (`population/factory/`), optics morphologies (`optics/factory/`), freezing models (`freezing/factory/`), and aerosol species (`species/`) expose a small, well-documented `build(...)` function (or use a decorator-based registry). At runtime, discovery maps the config `type` string to the appropriate builder. This keeps the public API small while enabling application-specific extensions without changing core code.

**Implication for practice.** The same downstream computations (e.g., CCN spectra, optical coefficients, freezing propensity) can be applied to many descriptions of aerosol particle populations. The current implementation includes modules for representing populations from PartMC, MAM4, or parameterized distributions, each defined with identical configurations. Because species properties and morphologies are provided through modular factories, sensitivity studies (e.g., refractive indices, mixing rules, or kappa values) become simple configuration changes rather than code modifications. This encourages transparent, process-level benchmarking across diverse datasets.

# Software description

## Design & architecture

The repository is organized around clear extension points:

- **`species/`** — The species registry provides canonical physical properties (e.g., density [kg m⁻³], kappa [–], molar mass [kg mol⁻¹], surface tension [N m⁻¹]) and a file/registry fallback. Public helpers include `register_species(...)`, `get_species(name, **mods)`, `list_species()`, and `retrieve_one_species(...)`. Per-species overrides (e.g., `{"SO4": {"kappa": 0.6}}`) apply at load time.

```python
from part2pop.species.registry import get_species
so4 = get_species("SO4", kappa=0.6)
```

* **`aerosol_particle`** — Defines the `Particle` class and helpers to build particles from species names, mass fractions, and diameters. A `Particle` stores per-species masses, dry/wet diameters, effective kappa, and basic metadata. Helpers provide kappa-Köhler growth and CCN activity [@Petters2007_ACP]. By default, CCN is treated with the homogeneous-sphere assumption and water surface tension.

```python
from part2pop.aerosol_particle import make_particle
p = make_particle(D=100e-9, aero_spec_names=["SO4"], aero_spec_frac=[1.0], D_is_wet=True)
print(p.get_Ddry(), p.get_tkappa())
```

* **`population/`** — Exposes `build_population(config)` and a discovery system mapping `config["type"]` to a module in `population/factory/`. The config settings are specific to the population type. For example, the `binned_lognormals` builder requires `"N"`, `"GMD"` (m), `"GSD"`, `"aero_spec_names"`, `"aero_spec_fracs"`, and binning parameters. The `partmc` builder, on the other hand, requires specification of the directory where simulation data is located and the simulation timestep. Each population-builder produces a standardized particle population, defined by a list of constituent `aerosol_species`, an array of `species_masses` that provide the mass [kg] of each species in each particle, an array of `num_conc` that describe the number concentration [m⁻3] associated with each computational particle, and a list of ids for each particle. An individual `Particle` is added to or extracted from the `ParticlePopulation` using the attached `add_particle` and `get_particle` functions, respectively. 

```python
from part2pop.population import build_population
pop = build_population({"type": "binned_lognormals", "N": [1e7], "GMD": [100e-9],
                        "GSD": [1.6], "aero_spec_names": [["SO4"]],
                        "aero_spec_fracs": [[1.0]], "N_bins": 120})
```

* **`optics/`** — `build_optical_population(pop, config)` attaches per-particle optical properties over `wvl_grid` (m) and `rh_grid` (default `[0.0]`). Morphologies (`homogeneous`, `core_shell`, `fractal`) compute per-particle scattering, absorption, and extinction cross-sections and asymmetry parameter `g`. The resulting `OpticalPopulation` aggregates to population-level coefficients (m⁻¹).

```python
from part2pop.optics import build_optical_population
opt_pop = build_optical_population(pop, {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]})
print(opt_pop.get_optical_coeff("b_scat", rh=0.0, wvl=550e-9))
```

* **`freezing/`** — Contains routines for assessing the potential for immersion freezing. The freezing module uses particle composition and surface area to calculate freezing proxies and exposes a builder pattern so new parameterizations can be added. Accepts `Particle` or `ParticlePopulation` inputs and returns particle-level and population metrics (e.g., activated fraction vs. T for a given cooling rate).
```python
from part2pop.freezing import build_freezing_population
freezing_pop = build_freezing_population(pop, {"morphology": "homogeneous", "T_grid": [-100,-80,-60,-40,-20,0], "T_units": "C"})
freezing_pop.get_frozen_fraction(-1.0) # specified cooling rate in K/s or C/s
```

* **`analysis/`** — Provides utilities for size distributions (`dN/dlnD`), moments, mass/volume fractions, hygroscopic growth factors, and CCN spectra. Returns NumPy arrays or lightweight dataclasses for plotting and statistics.

* **`viz/`** — Provides plotter builders, style management, and grid helpers for consistent visualization of population outputs.

*Implementation notes.* The codebase uses SI units internally (meters for diameters/wavelengths) and defaults `rh_grid` to `[0.0]`. Optional dependencies such as `netCDF4` or `PyMieScatt` are imported only where needed; in their absence the code raises `ModuleNotFoundError` with an actionable message rather than silently substituting mock data.

# Acknowledgements
The part2pop package was developed under the Integrated Cloud, Land-surface, and Aerosol System Study (ICLASS), a Science Focus Area of the U.S. Department of Energy's Atmospheric System Research program at Pacific Northwest National Laboratory (PNNL). Developoment of optics modules and links to the PartMC and MAM4 modules was supported by PNNL's Laboratory Directed Research and Development program. PNNL is a multi-program national laboratory operated for the U.S. Department of Energy by Battelle Memorial Institute under Contract No. DE-AC05-76RL01830.

# References

