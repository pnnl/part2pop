# Visualization

## What this module owns

- Plotter discovery and construction for visualization outputs.

## Public API entry points

- `build_plotter(type, config)`

## Plotter lifecycle

Plotters are lightweight objects built from configuration only. They do not
store population data or prepared plot data by default.

```text
source -> prep(source) -> prepared plot data -> render(prepared, ax)
```

- `prep(source)` computes plot-ready data from caller-provided source data.
- `render(prepared, ax)` draws already-prepared data onto a matplotlib axes.
- `plot(source, ax)` is the compatibility shortcut for `render(prep(source), ax)`.
- `plot(ax=ax, prepared=prepared)` renders prepared data and skips `prep()`.
- `plot_prepared(prepared, ax)` is an explicit prepared-data rendering helper.

`StatePlotter` sources are single particle populations. `SeriesPlotter` sources
are sequences or tables of population records.

## Examples

State plotters prepare one population and render an intrinsic state diagnostic:

```python
plotter = build_plotter("state_line", {"varname": "dNdlnD", "var_cfg": {}})
plotter.plot(population, ax)

prepared = plotter.prep(population)
plotter.plot(ax=ax, prepared=prepared)
```

Series plotters prepare many population records and render one scalar diagnostic
per record against an external x value:

```python
records = [
    {"population": pop0, "x": 0, "model": "partmc", "scenario": "001"},
    {"population": pop1, "x": 60, "model": "partmc", "scenario": "001"},
    {"population": pop2, "x": 0, "model": "mam4", "scenario": "001"},
    {"population": pop3, "x": 60, "model": "mam4", "scenario": "001"},
]

plotter = build_plotter("series_line", {
    "varname": "number_concentration",
    "x": "x",
    "series": "model",
    "group": "scenario",
    "population": "population",
    "var_cfg": {},
})
plotter.plot(records, ax)
```

Prepared series data can be rendered without population objects:

```python
prepared = {
    "data": [
        {"time_s": 0, "value": 1.0, "model": "partmc"},
        {"time_s": 60, "value": 2.0, "model": "partmc"},
    ],
    "x": "time_s",
    "y": "value",
    "series": "model",
    "xlabel": "time [s]",
    "ylabel": "number concentration [m$^{-3}$]",
}

plotter.plot(ax=ax, prepared=prepared)
```

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
