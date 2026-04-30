# ADR 0004: JOSS Release Scope

## Status

Accepted (release-defining)

## Context

- JOSS-facing work must prioritize a stable, testable, and extensible core package.
- Not all internal refactors are required to publish an accurate and useful paper/release.

## Decision

- JOSS release targets:
  - stable core package behavior
  - clear architecture and extension documentation
  - hardened registry/discovery behavior
  - strong public API and registry tests
- JOSS release does **not** require completing every future refactor.
- Observation builders (`EDX`, `HISCALE`) are currently functional but not fully internally reorganized.
- Partial Phase 2 progress is complete for observation assembly: `EDX` and `HISCALE` now terminate in shared `assemble_population_from_mass_fractions(...)` rather than routing explicit rows through `monodisperse`.
- Deeper internal cleanup for `EDX` / `HISCALE`, and broader refactors for `PartMC` / `MAM4`, remain deferred to Priority 2 unless time allows.
- JOSS paper may describe intended extension architecture without claiming every builder has already been internally reorganized.

Builder maturity (current):
- Distribution builders: stable/reference.
- Observation builders: functional, undergoing cleanup.
- Model builders: functional, deferred cleanup.

All builders produce valid ParticlePopulation objects and are covered by tests, even where internal organization is still evolving.

## Consequences

- Scope remains realistic and deliverable.
- Public claims align with implemented stability guarantees.
- Deferred work stays visible through roadmap and ADRs.
