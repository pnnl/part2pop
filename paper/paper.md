---
title: "part2pop: standardizing aerosol particle populations across models, parameterizations, and observations"
tags:
  - Python
  - aerosols
  - atmospheric science
  - aerosol microphysics
  - cloud condensation nuclei
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
---

# Summary

Aerosol particles are chemically diverse, multicomponent systems that span wide ranges of size, composition, and mixing state. These particle-scale differences strongly influence cloud condensation nuclei activity, ice nucleation, and optical properties [@Riemer2019_RG_MixingState; @Petters2007_ACP; @Cubison2008_ACP_CCN; @Fierce2013_JGR; @AlpertKnopf2016_ACP; @Lata2021_ACSESC_Freezing; @Fierce2016_NatComm; @Fierce2020_PNAS].

Across models, observations, and parameterized abstractions, aerosol populations are represented in incompatible ways. Particle-resolved models track per-particle composition, modal and sectional models represent distributions more compactly, reduced-order methods preserve selected moments or mixing-state information, and observation-based reconstructions combine instrument-specific measurements [@Riemer2009_JGR_PartMC; @Zaveri2008_JGR_MOSAIC; @Liu2016_GMD_MAM4; @Bauer2008_ACP_MATRIX; @Tilmes2023_GMD_CARMA; @Mann2012_ACP_modal_sectional; @Stevens2019_Atmosphere_MixingState; @Fierce2017_BAMS_mixing_state; @Fierce2017_JGR; @Fierce2024_JAS]. As a result, similar analyses are frequently reimplemented for each representation, making comparison and reuse time-consuming.

`part2pop` is a Python package that standardizes aerosol populations from heterogeneous sources into a common object model built around `AerosolSpecies`, `Particle`, and `ParticlePopulation`. The package supports population construction from model output, observation suites, and parametric descriptions, and it provides reusable particle-level and population-level diagnostics. By separating population representation from downstream analysis, `part2pop` enables interoperable workflows across otherwise incompatible aerosol descriptions.

# Statement of need

Aerosol populations are represented differently across particle-resolved models such as `PartMC-MOSAIC`, modal models such as `MAM4`, sectional models, reduced representations, and observation-constrained reconstructions [@Riemer2009_JGR_PartMC; @Zaveri2008_JGR_MOSAIC; @Liu2016_GMD_MAM4; @Bauer2008_ACP_MATRIX; @Tilmes2023_GMD_CARMA; @Mann2012_ACP_modal_sectional; @Stevens2019_Atmosphere_MixingState; @Zheng2021_ACP_MAM4_mixing_state; @Fierce2024_JAS]. Consequently, analyses such as CCN activity, hygroscopicity, composition metrics, optical properties, and freezing diagnostics are often implemented separately for each representation.

`part2pop` addresses this gap by providing a common aerosol-population interface that makes such analyses reusable across heterogeneous sources. The package is not a new aerosol simulation model. Instead, it standardizes populations derived from simulations, observations, and parameterized distributions so that they can be analyzed, compared, and reused consistently. Intended users include aerosol model developers, observational scientists reconstructing particle populations, downstream process modelers who require standardized aerosol inputs, and researchers performing intercomparisons or sensitivity analyses.

The need for `part2pop` is therefore not a lack of aerosol-physics software, but a lack of a shared aerosol population representation that allows those tools to be reused across models, parameterizations, and observations.

# State of the field

Existing tools in aerosol science typically focus on specific processes rather than representation interoperability. For example, `PyMieScatt` and `AeroMix` focus on optical-property calculations, while `pyrcel` focuses on cloud-parcel activation [@PyMieScatt; @Raj2024_GMD_AeroMix; @Rothenberg2017_GMD_pyrcel]. Upstream aerosol models such as `PartMC-MOSAIC`, `MAM4`, `MATRIX`, and `CARMA` generate aerosol populations, but in source-specific formats [@Riemer2009_JGR_PartMC; @Zaveri2008_JGR_MOSAIC; @Liu2016_GMD_MAM4; @Bauer2008_ACP_MATRIX; @Tilmes2023_GMD_CARMA]. Community efforts such as `GIANT` promote broader interface consistency among model components [@Hodzic2023_BAMS_GIANT].

`part2pop` complements rather than duplicates these tools. Its distinct contribution is a common population interface that accepts model-derived, parameterized, and observation-constrained aerosol populations and exposes them to shared diagnostics and extensions. This interface-level contribution matters because representation choices themselves can introduce structural error in aerosol diagnostics, especially for CCN-relevant quantities and reduced representations of mixing state [@Fierce2017_BAMS_mixing_state; @Fierce2017_JGR; @Fierce2024_JAS].

# Software design
![Overview of the config-driven architecture in `part2pop`. User-defined configuration files are processed by builder modules to construct standardized particle populations and derived analysis variables. Particle- and population-level diagnostics are then computed from these shared representations.\label{fig:overview}](overview_flowchart.png)

`part2pop` adopts a config-driven, object-oriented architecture centered on a small set of core data structures and registry-backed builder interface (Figure \ref{fig:overview}). The primary structures are `AerosolSpecies`, `ParticlePopulation`, and `Particle`. `AerosolSpecies` defines intrinsic species properties such as density, molecular weight, hygroscopicity, and refractive-index-related defaults, with optional overrides applied at build time. `ParticlePopulation` stores a species list, species-resolved particle masses, number concentrations, and particle identifiers. Individual particles are accessed and modified through `get_particle` and `set_particle`.

The core representation is mass-based. This is a deliberate design trade-off: it is general enough to accommodate heterogeneous population sources while remaining simple enough to support consistent derived-property calculations. Population construction is handled by `PopulationBuilder` using `population_cfg`, and diagnostic construction is handled by `VariableBuilder` using `var_cfg`. Both rely on registry-backed factories, so new population builders and diagnostic modules can be added without modifying the core classes.

Current population types include model-derived populations such as `partmc` and `mam4`, parameterized populations such as `monodisperse`, `binned_lognormals`, and `sampled_lognormals`, and observation-based populations such as `hiscale_observations`. The `hiscale_observations` builder reconstructs populations from complementary measurements associated with the HI-SCALE field campaign, including FIMS size distributions, AIMMS aircraft-state data, miniSPLAT composition classes, and AMS bulk-composition constraints [@Fast2019_BAMS_HISCALE; @Wang2017_JAS_FIMS1; @Wang2017_JAS_FIMS2; @Beswick2008_ACP_AIMMS20AQ; @Zelenyuk2015_JASMS_miniSPLAT; @Jayne2000_AST_AMS; @DeCarlo2006_AnalChem_HRToFAMS].

Process-specific extensions such as `OpticalPopulation` and `FreezingPopulation` wrap the same base population representation rather than duplicating population logic. This allows optical and freezing calculations to reuse the same underlying particle and population definitions while interfacing with established frameworks such as $\kappa$-Köhler theory and existing optics software [@Petters2007_ACP; @PyMieScatt; @AlpertKnopf2016_ACP]. A visualization module and optional graphical viewer support exploratory analysis without requiring users to define configuration files manually.

The architecture is also intentionally designed to be modifiable by both human developers and automated code-generation systems. Config-driven workflows, registry-based module discovery, minimal coupling between core data structures and downstream diagnostics, and stable builder interfaces localize code changes. As a result, new population builders or diagnostics can be added without invasive edits to the core package, lowering the barrier to extension for both human developers and AI-assisted code generation.

# Research impact statement

`part2pop` targets an active methodological bottleneck in aerosol science: representation choice itself can introduce structural error in aerosol diagnostics and can complicate comparison across models and datasets [@Fierce2017_BAMS_mixing_state; @Fierce2017_JGR; @Fierce2024_JAS]. By making particle-resolved, modal, sectional, parameterized, and observation-based populations analyzable through a shared interface, the package provides reusable infrastructure for representation-to-representation comparison, diagnostic reuse, and downstream model initialization.

The public release is accompanied by reproducible example workflows spanning optical-property calculations, cloud-relevant diagnostics, freezing-oriented analyses, and HI-SCALE observation-based populations. This positions `part2pop` as reusable research infrastructure rather than a study-specific analysis script and supports its use in intercomparison and observation-constrained aerosol analysis.

# AI usage disclosure

The core architecture, data model, and software design were developed by the authors without the use of generative AI tools. After the core structure was established, generative AI tools ([insert tool/model/version]) were used for code development and refactoring, development of the graphical user interface, and drafting and editing of documentation and manuscript text.

All AI-generated code and text were reviewed and modified by the authors, validated through testing and example workflows, and checked for scientific correctness. The software architecture was intentionally designed to support modification by automated code-generation tools, including large language models, through its modular and configuration-driven structure.

# Acknowledgements

The `part2pop` package was developed under the Integrated Cloud, Land-surface, and Aerosol System Study (ICLASS), a Science Focus Area of the U.S. Department of Energy Atmospheric System Research program at Pacific Northwest National Laboratory (PNNL). Development of optics modules and links to the `PartMC` and `MAM4` modules was supported by PNNL's Laboratory Directed Research and Development program. PNNL is a multi-program national laboratory operated for the U.S. Department of Energy by Battelle Memorial Institute under Contract No. DE-AC05-76RL01830.