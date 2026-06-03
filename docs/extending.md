# Extending part2pop

## Purpose

- Practical extension guide for contributors.
- Companion to:
  - `docs/roadmap.md`
  - `docs/architecture.md`
  - `docs/decisions/0001-builder-registry-contract.md`
  - `docs/decisions/0002-builder-categories-and-deferred-observation-model-work.md`

## Extension areas

- Population builders: `docs/population_builders.md`
- Analysis variables: `docs/analysis_variables.md`
- Optical morphologies: `docs/optics.md`
- Freezing parameterizations: `docs/freezing.md`
- Visualization plotters: `docs/visualization.md`
- Species and registry: `docs/species.md`

## Core extension rules

- Keep public APIs stable.
- Prefer decorator registration in factory modules.
- Keep fallback `module.build(...)` compatibility.
- Add tests for every new extension.
- Keep extension files small and focused.

## For AI coding assistants

- Keep PRs focused on one subsystem.
- Add tests for every new extension.
- Do not change public APIs without documenting it.
- Prefer small registered factory files.
- Do not add generated artifacts, local environment files, private scripts, or build metadata.
