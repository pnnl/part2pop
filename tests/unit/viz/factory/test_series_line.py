import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from part2pop import build_population
from part2pop.viz.builder import build_plotter
from part2pop.viz.factory.registry import discover_plotter_types
from part2pop.viz.factory.series_line import SeriesLinePlotter


def _population(total_number):
    weights = np.asarray([0.2, 0.3, 0.5]) * total_number
    return build_population(
        {
            "type": "monodisperse",
            "aero_spec_names": [["SO4"], ["SO4"], ["SO4"]],
            "N": weights.tolist(),
            "D": [50e-9, 100e-9, 200e-9],
            "aero_spec_fracs": [[1.0], [1.0], [1.0]],
        }
    )


def _records():
    return [
        {"population": _population(1.0e6), "x": 60, "model": "partmc", "scenario": "001"},
        {"population": _population(2.0e6), "x": 0, "model": "partmc", "scenario": "001"},
        {"population": _population(3.0e6), "x": 60, "model": "mam4", "scenario": "001"},
        {"population": _population(4.0e6), "x": 0, "model": "mam4", "scenario": "001"},
    ]


def _nccn_config(**overrides):
    cfg = {"varname": "Nccn", "var_cfg": {"s_grid": [1.0]}}
    cfg.update(overrides)
    return cfg


def test_series_line_is_discoverable_and_builds():
    types = discover_plotter_types()

    assert "series_line" in types
    assert isinstance(build_plotter("series_line", _nccn_config()), SeriesLinePlotter)


def test_series_line_prep_computes_y_values_with_real_variable():
    plotter = SeriesLinePlotter(_nccn_config(series="model", group="scenario"))

    prepared = plotter.prep(_records())

    assert prepared["x"] == "x"
    assert prepared["y"] == "y"
    assert prepared["series"] == "model"
    assert prepared["group"] == "scenario"
    assert [row["y"] for row in prepared["data"]] == [1.0e6, 2.0e6, 3.0e6, 4.0e6]
    assert "population" not in prepared["data"][0]
    assert prepared["ylabel"] == "CCN number concentration [m$^{-3}$]"


def test_series_line_plot_records_draws_one_line_per_series():
    plotter = SeriesLinePlotter(_nccn_config(series="model"))
    fig, ax = plt.subplots()

    plotter.plot(_records(), ax)

    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {"partmc", "mam4"}


def test_series_line_plot_prepared_does_not_require_populations_or_varname():
    plotter = SeriesLinePlotter({"series": "model"})
    prepared = {
        "data": [
            {"time_s": 0, "value": 1.0, "model": "partmc"},
            {"time_s": 60, "value": 2.0, "model": "partmc"},
            {"time_s": 0, "value": 3.0, "model": "mam4"},
            {"time_s": 60, "value": 4.0, "model": "mam4"},
        ],
        "x": "time_s",
        "y": "value",
        "series": "model",
        "xlabel": "time [s]",
        "ylabel": "number concentration [m$^{-3}$]",
        "xscale": "linear",
        "yscale": "linear",
    }
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "time [s]"
    assert ax.get_ylabel() == "number concentration [m$^{-3}$]"


def test_series_line_plot_prepared_method():
    plotter = SeriesLinePlotter({"group": "scenario"})
    prepared = {
        "data": [
            {"x": 0, "y": 1.0, "scenario": "001"},
            {"x": 60, "y": 2.0, "scenario": "001"},
        ],
        "x": "x",
        "y": "y",
        "group": "scenario",
    }
    fig, ax = plt.subplots()

    plotter.plot_prepared(prepared, ax)

    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "001"


def test_series_line_group_creates_separate_lines():
    plotter = SeriesLinePlotter({"group": "scenario"})
    prepared = {
        "data": [
            {"x": 0, "y": 1.0, "scenario": "001"},
            {"x": 60, "y": 2.0, "scenario": "001"},
            {"x": 0, "y": 3.0, "scenario": "002"},
            {"x": 60, "y": 4.0, "scenario": "002"},
        ],
        "x": "x",
        "y": "y",
        "group": "scenario",
    }
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {"001", "002"}


def test_series_line_series_and_group_create_lines_per_pair():
    plotter = SeriesLinePlotter({"series": "model", "group": "scenario"})
    prepared = {
        "data": [
            {"x": 0, "y": 1.0, "model": "partmc", "scenario": "001"},
            {"x": 60, "y": 2.0, "model": "partmc", "scenario": "001"},
            {"x": 0, "y": 3.0, "model": "partmc", "scenario": "002"},
            {"x": 60, "y": 4.0, "model": "partmc", "scenario": "002"},
            {"x": 0, "y": 5.0, "model": "mam4", "scenario": "001"},
            {"x": 60, "y": 6.0, "model": "mam4", "scenario": "001"},
        ],
        "x": "x",
        "y": "y",
        "series": "model",
        "group": "scenario",
    }
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert len(ax.lines) == 3
    assert {line.get_label() for line in ax.lines} == {
        "partmc / 001",
        "partmc / 002",
        "mam4 / 001",
    }


def test_series_line_sorts_rows_by_x():
    plotter = SeriesLinePlotter({"sort": True})
    prepared = {
        "data": [
            {"x": 60, "y": 2.0},
            {"x": 0, "y": 1.0},
        ],
        "x": "x",
        "y": "y",
    }
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert np.asarray(ax.lines[0].get_xdata()).tolist() == [0, 60]
    assert np.asarray(ax.lines[0].get_ydata()).tolist() == [1.0, 2.0]


def test_series_line_vector_y_raises_with_real_variable():
    records = [{"population": _population(1.0e6), "x": 0}]
    edges = np.asarray([1.0e-8, 8.0e-8, 1.5e-7, 3.0e-7])
    plotter = SeriesLinePlotter(
        {
            "varname": "dNdlnD",
            "var_cfg": {"method": "hist", "edges": edges, "diam_grid": np.sqrt(edges[:-1] * edges[1:])},
        }
    )

    with pytest.raises(ValueError, match="series_line requires scalar y values"):
        plotter.prep(records)


def test_series_line_applies_label_and_scale_fallbacks():
    plotter = SeriesLinePlotter(
        {
            "xlabel": "configured x",
            "ylabel": "configured y",
            "xscale": "linear",
            "yscale": "log",
        }
    )
    prepared = {"data": [{"x": 1, "y": 10}], "x": "x", "y": "y"}
    fig, ax = plt.subplots()

    plotter.plot(ax=ax, prepared=prepared)

    assert ax.get_xlabel() == "configured x"
    assert ax.get_ylabel() == "configured y"
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "log"
