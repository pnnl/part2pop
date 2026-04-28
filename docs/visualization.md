# Visualization

## What this module owns

- Plotter discovery and construction for visualization outputs.

## Public API entry points

- `build_plotter(type, config)`

## List/describe registered components

- `list_plotter_types()`
- `describe_plotter_type(name)`

## How to add a new extension

1. Add a factory module under `src/part2pop/viz/factory/`.
2. Provide `build(config)` for the plotter.
3. Prefer `@register("name")` registration.

## Required file location and naming pattern

- Location: `src/part2pop/viz/factory/`
- Pattern: one plotter per module, lowercase filename, no leading `_`.

## Minimal code example

```python
from part2pop.viz.factory.registry import register

@register("my_plotter")
def build(config):
    """Return a plotter instance."""
    return object()
```

## Recommended tests

- `list_plotter_types()` includes new entry.
- `describe_plotter_type("my_plotter")` returns metadata.
- `build_plotter("my_plotter", config)` constructs successfully.

## What not to do / deferred work

- Do not treat `viewer/metadata.py` as long-term metadata source of truth.
- GUI metadata migration remains staged work.
