# Freezing

## What this module owns

- Freezing parameterization builders for particle/population freezing diagnostics.

## Public API entry points

- `build_freezing_particle(base_particle, config)`
- `build_freezing_population(base_population, config)`

## List/describe registered components

- `list_freezing_types()`
- `describe_freezing_type(name)`

## Naming note

- Some legacy code paths still use the term `morphology` for freezing type selection.

## How to add a new extension

1. Add a module under `src/part2pop/freezing/factory/`.
2. Implement `build(base_particle, config)`.
3. Prefer `@register("name")` for registration.

## Required file location and naming pattern

- Location: `src/part2pop/freezing/factory/`
- Pattern: one type per module, lowercase filename, no leading `_`.

## Minimal code example

```python
from part2pop.freezing.factory.registry import register

@register("my_freezing_type")
def build(base_particle, config):
    """Return a freezing wrapper/parameterization instance."""
    return base_particle
```

## Recommended tests

- `list_freezing_types()` includes new entry.
- `describe_freezing_type("my_freezing_type")` returns metadata.
- `build_freezing_particle(..., {"morphology": "my_freezing_type"})` works.

## What not to do / deferred work

- No freezing architecture overhaul in this PR.
