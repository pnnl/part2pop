# ADR 0001: Builder Registry Contract

## Status

Accepted (JOSS-facing baseline)

## Context

- part2pop exposes multiple builder categories via module-level and registry-driven entry points.
- JOSS-facing stabilization requires predictable discovery and stable public APIs.

## Decision

- Registered components should be discoverable.
- Broken optional components should not break unrelated discovery.
- Decorator-based registration is preferred.
- Module-level `build(...)` fallback may remain for backward compatibility.
- Registries should evolve to expose `list_*` and `describe_*` functions.
- Population builders now commonly terminate in a shared population assembly helper for explicit particle rows.
- Public builder APIs remain stable:
  - `build_population(config)`
  - `build_variable(...)`
  - `build_optical_particle(...)`
  - `build_optical_population(...)`
  - `build_freezing_particle(...)`
  - `build_freezing_population(...)`
  - `build_plotter(...)`

## Consequences

- Contributors can add components with a predictable contract.
- Discovery and metadata can serve docs, GUI integration, and AI tooling.
- Backward compatibility is preserved while modern registry behavior is hardened.
