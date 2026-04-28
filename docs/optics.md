# Optics

## What this module owns

- Optical morphologies and optical particle/population wrappers.

## Public API entry points

- `build_optical_particle(base_particle, config)`
- `build_optical_population(base_population, config)`

## List/describe registered components

- `list_morphology_types()`
- `describe_morphology_type(name)`

## How to add a new extension

1. Add a factory module in `src/part2pop/optics/factory/`.
2. Provide callable `build(base_particle, config)`.
3. Prefer `@register("name")`.

## Required file location and naming pattern

- Location: `src/part2pop/optics/factory/`
- Pattern: one morphology per module, lowercase filename, no leading `_`.

## Minimal code example

```python
from part2pop.optics.factory.registry import register

@register("my_morphology")
def build(base_particle, config):
    """Return an optical particle wrapper."""
    return base_particle
```

## Recommended tests

- `list_morphology_types()` includes the new entry.
- `describe_morphology_type("my_morphology")` returns metadata.
- `build_optical_particle(..., {"type": "my_morphology"})` works.

## What not to do / deferred work

- No broad optics architecture refactor in this PR.
