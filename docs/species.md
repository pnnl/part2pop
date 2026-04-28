# Species

## What this module owns

- `AerosolSpecies` representation.
- Runtime species registration.
- File-backed fallback species retrieval.

## Public API entry points

- `register_species(species)`
- `get_species(name, specdata_path, **modifications)`
- `list_species()`
- `describe_species(name)`
- `extend_species(species)`

## List/describe registered components

- `list_species()` currently lists custom runtime-registered species only.
- `describe_species(name)` supports:
  - custom registered species
  - default file-backed species (if found)

## How to add a new extension

1. Create an `AerosolSpecies` object.
2. Register it with `register_species(...)`.
3. Use it through builders or `get_species(...)`.

## Required file location and naming pattern

- Runtime registration: no file required.
- Default file-backed species data source:
  - `species_data/aero_data.dat` via package data loader.

## Minimal code example

```python
from part2pop.species import AerosolSpecies, register_species, describe_species

spec = AerosolSpecies(
    name="MY_SPEC",
    density=1500.0,
    kappa=0.2,
    molar_mass=100.0,
    surface_tension=0.072,
)
register_species(spec)
print(describe_species("MY_SPEC"))
```

## Recommended tests

- Register/list round-trip.
- Describe custom species.
- Describe known default species.
- Unknown species error behavior.

## specdata_path notes

- `get_species(..., specdata_path=...)` can point to an alternate species data folder.
- If `specdata_path` is `None`, package default data is used.

## What not to do / deferred work

- Do not introduce a new species metadata schema in this phase.
