# part2pop Architecture (JOSS-facing)

## Goals

- Keep public API simple and stable.
- Make extension points discoverable and safe.
- Support humans, GUI code, and AI assistants via clear metadata/docs.

## Core data model

- `AerosolSpecies`
- `Particle`
- `ParticlePopulation`

These are the core aerosol representation primitives and remain central to package behavior.

## Builder/registry extension surface

part2pop provides builder entry points and registries for:

- population builders
- analysis variables
- optical morphologies
- freezing parameterizations
- visualization plotters

Related module families:

- `part2pop.population`
- `part2pop.analysis`
- `part2pop.optics`
- `part2pop.freezing`
- `part2pop.viz`
- `part2pop.species`

## Intended contract (current direction)

- Public builder APIs remain stable and easy to call.
- Registration should prefer decorators.
- Discovery should be resilient to optional/degraded components.
- Backward compatibility with module-level `build(...)` remains where feasible.
- Registry surfaces should evolve toward package-level `list_*` / `describe_*` APIs.

## Discovery and metadata

- Users should be able to enumerate available components by category.
- Metadata should be structured enough for:
  - docs
  - GUI configuration
  - automation/AI-assisted workflows
- Long-term direction: metadata originates from package-level describe/list APIs, not duplicated hard-coded tables.

## JOSS release architecture scope

### In scope (Priority 1)

- Stabilize core extension architecture and discovery behavior.
- Document extension points and contracts.
- Expand public API and registry tests.

### Current status and deferred follow-up

- Observation-based builders (`edx_observations`, `hiscale_observations`) now include helper-layer cleanup/splitting in place.
- Model-derived builders:
  - `partmc` row-preparation cleanup was completed in place.
  - `mam4` remains intentionally thin and delegates to `binned_lognormals`.
- Species-name alias resolution currently applies to direct species-list builders `monodisperse` and `binned_lognormals`.
- EDX elemental-to-species reconstruction clarification remains a follow-up issue and is not part of JOSS release finalization.
