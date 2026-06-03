# Analysis Variables

## What this module owns

- Particle-level and population-level diagnostic variables.
- Variable registration/discovery and metadata (`VariableMeta`).

## Public API entry points

- `build_variable(...)`
- analysis variable builders under `src/part2pop/analysis/*/factory/`

## List/describe registered components

- Use existing analysis registry list/describe APIs where available.
- Pattern aligns with other extension systems: list first, then describe selected variable.

## How to add a new extension

1. Choose scope:
   - particle variable
   - population variable
2. Add a factory module in the corresponding analysis factory location.
3. Define builder and attach `VariableMeta`.
4. Ensure it is discoverable by existing analysis registry logic.

## Required file location and naming pattern

- Location: `src/part2pop/analysis/particle/factory/` or `src/part2pop/analysis/population/factory/`
- Pattern: one variable per module, lowercase filename, no leading `_`.

## Minimal code example

```python
from part2pop.analysis.base import VariableMeta

meta = VariableMeta(
    name="my_var",
    axis_names=("particle",),
    description="Example variable",
)

def build(cfg=None):
    obj = type("Var", (), {})()
    obj.meta = meta
    return obj
```

## Recommended tests

- Builder returns variable object with `meta`.
- Describe API returns expected metadata.
- Variable can be built through `build_variable(...)`.

## What not to do / deferred work

- Do not introduce a new formal metadata schema in this PR.
- Keep analysis changes focused on extension discoverability and tests.
