# Population Builders

## What this module owns

- Construction of `ParticlePopulation` from config dictionaries.
- Discovery of population builder implementations under factory modules.
- Population builders live in `src/part2pop/population/factory/`.

## Public API entry points

- `build_population(config)`
- `PopulationBuilder(config).build()`

## Builder contract

- A population builder is a **registered callable**.
- Signature: accepts `config` and returns `ParticlePopulation`.
- `PopulationBuilder` is a dispatcher/facade, not a required superclass.

## List/describe registered components

- `list_population_types()`
- `describe_population_type(name)`

## Builder categories

- Direct species-list / distribution builders:
  - `monodisperse`
  - `binned_lognormals`
  - `sampled_lognormals`
- Observation-based builders:
  - `edx_observations`
  - `hiscale_observations`
- Model-derived builders:
  - `partmc`
  - `mam4`

## How to add a new extension

1. Add a new factory file under:
   - `src/part2pop/population/factory/<name>.py`
2. Implement `build(config)` that returns a `ParticlePopulation`.
3. Prefer decorator registration via `@register("<name>")`.
4. Keep module-level `build(...)` available for fallback compatibility.

## Required file location and naming pattern

- Location: `src/part2pop/population/factory/`
- Pattern: one builder per module, lowercase filename, no leading `_`.
- Simple builders can remain self-contained in a single factory module.
- Larger builders may keep internal support code in:
  - `src/part2pop/population/factory/helpers/`
- `helpers/` is internal and is **not** a builder plugin location.

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
