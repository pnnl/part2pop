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
- Release metadata source of truth is `pyproject.toml` (version and package-facing project metadata).
- Legacy setuptools metadata must not conflict with the JOSS release version (`1.0.0`) or project description.
- User/reviewer installation docs must include a source-install path for repository-state evaluation, independent of PyPI publication timing.
- JOSS release does **not** require completing every future refactor.
- Observation builders (`EDX`, `HISCALE`) are currently functional but not fully internally reorganized.
- Partial Phase 2 progress is complete for observation assembly: `EDX` and `HISCALE` now terminate in shared `assemble_population_from_mass_fractions(...)` rather than routing explicit rows through `monodisperse`.
- Deeper internal cleanup for `EDX` / `HISCALE`, and broader refactors for `PartMC` / `MAM4`, remain deferred to Priority 2 unless time allows.
- `ParticlePopulation.reduce_mixing_state(...)` is intentionally unsupported for the JOSS 1.0.0 release pending scientific specification and validation; it should fail explicitly with `NotImplementedError`, not incidental runtime errors.
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
- Submit-stopper cleanup expectations are explicit for metadata consistency, install guidance, and intentionally deferred behavior.
