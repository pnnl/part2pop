# part2pop JOSS Roadmap

## Scope of this document

- Planning/documentation guide for JOSS-facing stabilization.
- This roadmap is intentionally **not** a source-code refactor PR.
- Observation-based and model-derived builder refactors are explicitly deferred to Priority 2.

## Priority 1 — must be done before JOSS

### 1) Preserve public APIs

- Keep stable and documented:
  - `Particle`
  - `AerosolSpecies`
  - `ParticlePopulation`
  - `build_population(config)`
  - `build_variable(...)`
  - `build_optical_particle(...)`
  - `build_optical_population(...)`
  - `build_freezing_particle(...)`
  - `build_freezing_population(...)`
  - `build_plotter(...)`

### 2) Harden registry/factory discovery

- Harden discovery behavior across:
  - `population`
  - `analysis`
  - `optics`
  - `freezing`
  - `viz`
- Ensure registries are:
  - decorator-aware
  - safe against optional-component import failure
  - discoverable and testable
- Preserve backward compatibility with module-level `build(...)` where feasible.

### 3) Add or plan list/describe metadata APIs

- Ensure package-level discoverability for:
  - population types
  - analysis variables
  - optical morphologies
  - freezing parameterizations
  - plotters
  - species
- Target API style:
  - `list_*()` for names/keys
  - `describe_*()` for machine- and human-readable metadata

### 4) Document extension points

- Add module-level extension docs so contributors and AI tools can extend consistently.
- Keep examples concise and aligned with actual registry contracts.

### 5) Test and release hardening

- Add/expand tests for:
  - public API stability
  - registry discovery and resilience
  - list/describe metadata interfaces (as they land)
- Coverage target:
  - keep codecov > 90%
  - move toward 95%+ where meaningful
- Remove release-unsafe clutter:
  - generated artifacts
  - stale debug files
  - local IDE leftovers

### 6) Document JOSS release scope

- Clearly state what this release guarantees.
- Explicitly state deferred work (Priority 2+).

## Priority 2 — good before JOSS, but not required

### Observation-based builder refactor (deferred)

- `EDX`
- `HISCALE`
- Introduce explicit `reconstruction_strategy` concept.
- Target pathway:
  - reader → reconstruction strategy → population assembly

### Model-derived builder cleanup (deferred)

- `PartMC`
- `MAM4`
- Clarify what belongs in `part2pop` vs `AMBRS`.

### Shared assembly and metadata improvements

- Add shared population assembly helper / `PopulationInputs` if needed.
- Add richer metadata for GUI support.
- Add IO format docs and roundtrip tests if still needed after Priority 1.

## Priority 3 — future-facing wishlist

- Full HISCALE refactor into reader / selector / reconstruction / assembly.
- Full EDX refactor with multiple reconstruction strategies.
- MAM4 and PartMC translator cleanup.
- AMBRS adapter cleanup.
- PARCI interface cleanup.
- GUI migration away from hard-coded `viewer/metadata.py`.
- Formal config schema / `ConfigField` / `BuilderMeta` system.
- Migrate distribution builders to shared assembly helpers.
- Optional dependency cleanup in `pyproject.toml`.
- Full docs site.
- 100% coverage only where meaningful.

## Suggested PR sequence (Priority 1 execution plan)

1. **PR 1: roadmap and architecture docs**
   - Add roadmap, architecture, and decision records.
2. **PR 2: registry/discovery hardening**
   - Stabilize decorator-aware registration and safe discovery behavior.
3. **PR 3: list/describe metadata APIs**
   - Introduce discoverability APIs across extension categories.
4. **PR 4: module-level extension docs**
   - Add practical extension docs in core modules.
5. **PR 5: public API and registry test hardening**
   - Lock behavior with tests and maintain coverage targets.
6. **PR 6: release hygiene and JOSS readiness pass**
   - Cleanup artifacts, verify docs, run final checks.
7. **Priority 2 follow-up**
   - Observation/model-derived builder refactors (`EDX`, `HISCALE`, `PartMC`, `MAM4`).
