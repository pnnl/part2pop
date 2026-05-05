# part2pop JOSS Roadmap

## Scope of this document

- Planning/documentation guide for JOSS-facing stabilization.
- This roadmap is a living document and should be updated as focused cleanup PRs are completed.
- This roadmap is intentionally **not** a source-code refactor PR.
- It records:
  - completed JOSS-prep work,
  - completed and deferred builder-architecture work,
  - future-facing wishlist items.

## Maintenance note

Update this file after each focused PR or meaningful design decision.

Each update should briefly reflect:

- what was completed,
- what is currently active,
- what was intentionally deferred,
- whether any item moved between priorities.

Cline should periodically update this file as part of PR wrap-up summaries, but should not use this document as permission to broaden the scope of a focused source-code PR.

---

## Completed Phase 1 — JOSS-facing stabilization

Phase 1 focused on release safety, public API stability, registry/factory clarity, and documentation needed before moving into deeper population-builder cleanup.

### Public API preservation

Stable public entry points were preserved and should remain documented:

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

### Registry/factory discovery hardening

Registry and factory behavior was clarified across:

- `population`
- `analysis`
- `optics`
- `freezing`
- `viz`
- `species`

Important design decision:

- `PopulationBuilder` is a dispatcher/facade.
- Population builders are registered callables.
- Population builders do **not** need to be subclasses/classes.

### List/describe metadata APIs

Discovery APIs were added or planned for extension categories using the general style:

- `list_*()` for available names/keys
- `describe_*()` for machine- and human-readable metadata

### Extension documentation

Module-level extension documentation was added or improved so contributors and AI tools can extend the package consistently.

Extension surfaces include:

- population builders
- analysis variables
- optical morphologies
- freezing parameterizations
- visualization plotters
- species

### Test and release hardening

Phase 1 added or expanded tests for:

- public API stability
- registry discovery and resilience
- metadata/list/describe behavior
- release-safe behavior around optional components

Coverage target remains:

- keep codecov above 90%
- move toward 95%+ where meaningful

### Release-scope documentation

The JOSS-facing release scope was clarified.

Deferred work is tracked below rather than being mixed into the release-stabilization checklist.

### Deferred behavior for JOSS 1.0.0

- `ParticlePopulation.reduce_mixing_state(...)` is intentionally unsupported in
  the 1.0.0 JOSS release.
- The legacy implementation was removed because it could fail with incidental
  runtime errors instead of a scientifically validated behavior.
- A future release may reintroduce this capability once behavior is specified
  and validated.

---

## Population builder cleanup status (post-Phase-2 implementation pass)

Phase 2 focuses on cleaning up population-builder architecture, especially observation- and model-derived builders, while preserving public APIs.

Current architectural direction:

```text
ParticlePopulation = canonical in-memory product
build_population(config) = public config entry point
population/factory/*.py = registered builder entry points
population/factory/helpers/* = internal reusable support code
```

Preferred folder shape:

```text
population/
  builder.py
  base.py
  factory/
    registry.py
    monodisperse.py
    binned_lognormals.py
    sampled_lognormals.py
    edx_observations.py
    hiscale_observations.py
    partmc.py
    mam4.py

    helpers/
      assembly.py
      validation.py
      species_alignment.py
      species_resolution.py
      edx.py
      hiscale.py
      model_translation.py
```

### Completed implementation outcomes

Completed:

* Internal helper added/stabilized:

```text
src/part2pop/population/factory/helpers/assembly.py
```

Target helper:

```python
assemble_population_from_mass_fractions(...)
```

Purpose:

* Convert explicit particle rows into a canonical `ParticlePopulation`.
* Align local per-particle species/fraction rows to a population-wide species list.
* Preserve the expected shape contract:

```text
Particle.masses: 1D, length n_species
ParticlePopulation.spec_masses: 2D, shape (n_particles, n_species)
ParticlePopulation.num_concs: 1D, shape (n_particles,)
```

Implemented design:

* `hiscale_observations` should **not** route explicit particle rows through `monodisperse`.
* `hiscale_observations` should call the internal assembly helper directly once it has produced:

  * `particle_diameters`
  * `particle_num_concs`
  * `aero_spec_names`
  * `aero_spec_fracs`

Resolved shape-regression target:

```text
ParticlePopulation.spec_masses becomes shape (1000, 1, 3)
Expected shape is (1000, 3)
```

Outcome:

* Diagnose with targeted shape instrumentation before patching.
* Do not blindly change:

  * `Particle`
  * `make_particle`
  * `ParticlePopulation`
  * `population/base.py`

### Observation-based builders

Builders:

* `EDX`
* `HISCALE`

Target pathway:

```text
reader
→ source-specific parsing
→ reconstruction strategy
→ species resolution
→ explicit particle rows
→ shared population assembly
```

Current status:

* HISCALE direct-to-assembler wiring is in place.
* EDX/HISCALE helper-layer cleanup/splitting is in place for current release scope.
* Additional EDX elemental-to-species reconstruction clarification remains a follow-up item.

### Shared assembly and metadata improvements

Completed/near-term:

* Add shared population assembly helper.
* Add focused tests for helper shape contracts.
* Add regression tests only at the layer proven to be responsible for shape bugs.

Deferred:

* `PopulationInputs` dataclass or formal internal input object.
* Richer GUI metadata.
* IO format docs and roundtrip tests, if still needed.

### Species resolution for population builders

Add an internal species-resolution layer for observation- and model-derived population builders.

Goal:

* Translate source-specific species labels into canonical `part2pop` species names before population assembly.

Examples:

* `"dust"`, `"Dust"` → `"OIN"`
* `"soot"`, `"BC-containing"`, `"black carbon"` → `"BC"`
* `"POA"` → `"OC"` or an explicitly configured organic surrogate
* IEPOX-related components → either mapped to an existing organic surrogate or defined with explicit species properties

Target pathway:

```text
source labels
→ canonical species names
→ known/default species properties
→ population assembly
```

This should happen before:

```python
assemble_population_from_mass_fractions(...)
```

so the assembler receives canonical species names with known properties.

Important boundary:

* Species resolution should not be solved as part of the current HISCALE shape-regression bugfix.
* The assembler should primarily assemble already-resolved species rows.
* Builder-specific source labels should be resolved upstream of assembly.

### Model-derived builders

Builders:

* `PartMC`
* `MAM4`

Target pathway:

```text
model-native representation
→ model translation helper
→ species resolution
→ explicit particle rows
→ shared population assembly
```

Current status / deferred questions:

* `partmc` row-preparation cleanup was completed in place.
* `mam4` was inspected and remains intentionally thin because it delegates to `binned_lognormals`.

* What belongs in `part2pop`?
* What belongs in AMBRS?
* How much model-specific translation should live in `population/factory/helpers/model_translation.py`?

---

## Priority 3 — future-facing wishlist

* Full HISCALE refactor into:

  * reader
  * selector
  * reconstruction strategy
  * species resolution
  * assembly
* Full EDX refactor with multiple reconstruction strategies.
* MAM4 and PartMC translator cleanup.
* AMBRS adapter cleanup.
* PARCI interface cleanup.
* GUI migration away from hard-coded `viewer/metadata.py`.
* Formal config schema / `ConfigField` / `BuilderMeta` system.
* Introduce a clearly named explicit-row builder, such as:

  * `explicit_population`
  * `discrete_distribution`
  * `discrete_particles`
* Decide whether legacy `monodisperse` should become a one-row special case of the explicit-row builder.
* Migrate distribution builders to shared assembly helpers after their mode-to-row logic is stable:

  * `binned_lognormals`
  * `sampled_lognormals`
* Optional dependency cleanup in `pyproject.toml`.
* Full docs site.
* 100% coverage only where meaningful.

---

## Current suggested PR sequence

### Completed

1. **Phase 1 PRs: roadmap, architecture docs, registry/discovery hardening, metadata APIs, extension docs, public API tests, release hygiene**

   * Status: completed or largely completed.
   * Outcome: Phase 1 is considered complete.

### Active Phase 2

2. **PR: Add/stabilize explicit-row assembly helper**

   * Add `population/factory/helpers/assembly.py`.
   * Add helper-level tests for local-row species/fraction alignment.
   * Keep helper internal for now.

3. **PR: Refactor HISCALE final assembly path**

   * Keep public `build_population({"type": "hiscale_observations", ...})`.
   * Do not route final explicit rows through `monodisperse`.
   * Call `assemble_population_from_mass_fractions(...)` directly.
   * Diagnose and fix the current `(1000, 1, 3)` `spec_masses` regression with evidence.

4. **PR: Refactor EDX final assembly path**

   * Use the same explicit-row helper if EDX already produces particle rows.
   * Add narrow tests around EDX row assembly and shape contracts.

### Later Phase 2 / Phase 3

5. **PR: Species resolution helper**

   * Add internal alias/canonicalization support.
   * Apply first to observation-derived builders.
   * Keep separate from shape-regression fixes.

6. **PR: Model-derived builder cleanup**

   * Clarify MAM4 and PartMC translation responsibilities.
   * Add shared model-translation helper only if needed.

7. **PR: Explicit/discrete population builder design**

   * Decide whether to introduce a new registered builder for explicit rows.
   * Decide whether `monodisperse` remains legacy behavior or becomes a one-row wrapper.

---

## Cline workflow expectations

Cline should keep this roadmap current, but must not use it to expand PR scope.

For each focused PR, Cline should report whether `docs/roadmap.md` needs an update.

Roadmap edits are appropriate when:

* a Phase 2 item is completed,
* a design decision is made,
* an item is moved between active and deferred,
* a new TODO is identified,
* a previous TODO is intentionally rejected.

Roadmap edits are not appropriate when:

* the PR only fixes a typo,
* the PR only adjusts test mechanics without changing project direction,
* the roadmap change would broaden an otherwise focused bugfix.
