# ADR 0002: Builder Categories and Deferred Observation/Model Work

## Status

Accepted (JOSS-facing scope control)

## Context

- part2pop includes multiple builder types with different maturity and complexity.
- JOSS release readiness depends on stabilizing core extension infrastructure first.

## Decision

### Distribution builders (Priority 1 baseline)

- `monodisperse`
- `binned_lognormals`
- `sampled_lognormals`

### Observation-constrained builders (deferred to Priority 2)

- `EDX`
- `HISCALE`
- Current implementation status:
  - both now terminate in shared `assemble_population_from_mass_fractions(...)`
  - both are functional, but internal cleanup/reorganization is still pending
- Future direction:
  - introduce explicit `reconstruction_strategy`
  - use reader → reconstruction strategy → population assembly flow

### Model-derived builders (deferred to Priority 2)

- `PartMC`
- `MAM4`
- Future direction:
  - clarify ownership split between `part2pop` and `AMBRS`

## Consequences

- Priority 1 work stays focused on extension system stability, docs, and tests.
- JOSS release does not require full internal reorganization of all builders.
- Deferred categories remain documented and planned, not blocked or hidden.
