# Population Builders

## What this module owns

- Construction of `ParticlePopulation` from config dictionaries.
- Discovery of population builder implementations under factory modules.

## Public API entry points

- `build_population(config)`
- `PopulationBuilder(config).build()`

## List/describe registered components

- `list_population_types()`
- `describe_population_type(name)`

## Builder categories

- Distribution builders (current Priority 1 baseline):
  - `monodisperse`
  - `binned_lognormals`
  - `sampled_lognormals`
- Observation-constrained builders (deeper refactor deferred to Priority 2):
  - `EDX`
  - `HISCALE`
- Model-derived builders (cleanup deferred to Priority 2):
  - `PartMC`
  - `MAM4`

## How to add a new extension

1. Add a new factory file under:
   - `src/part2pop/population/factory/<name>.py`
2. Implement `build(config)` that returns a `ParticlePopulation`.
3. Prefer decorator registration via `@register("<name>")`.
4. Keep module-level `build(...)` available for fallback compatibility.

## Required file location and naming pattern

- Location: `src/part2pop/population/factory/`
- Pattern: one builder per module, lowercase filename, no leading `_`.

## Minimal code example

```python
from part2pop.population.factory.registry import register
from part2pop.population.base import ParticlePopulation

@register("my_population")
def build(config):
    """Build a ParticlePopulation from config."""
    pop = ParticlePopulation(species=[], masses=[], num_concs=[], ids=[])
    return pop
```

## Recommended tests

- Factory discovery test: builder appears in `list_population_types()`.
- Describe test: `describe_population_type("my_population")` returns metadata.
- Builder test: `build_population({"type": "my_population", ...})` works.

## What not to do / deferred work

- Do not refactor EDX/HISCALE internals in this phase.
- Do not refactor PartMC/MAM4 internals in this phase.
- Do not introduce formal schema systems yet.
