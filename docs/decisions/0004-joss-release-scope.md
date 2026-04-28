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
- Deeper refactors for `EDX` / `HISCALE` and `PartMC` / `MAM4` are deferred to Priority 2 unless time allows.
- JOSS paper may describe intended extension architecture without claiming every builder has already been internally reorganized.

## Consequences

- Scope remains realistic and deliverable.
- Public claims align with implemented stability guarantees.
- Deferred work stays visible through roadmap and ADRs.
