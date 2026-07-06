import importlib

import pytest

from part2pop.viz.factory import registry as viz_registry


def test_import_viz_factory_registry():
    importlib.import_module("part2pop.viz.factory.registry")


def test_discover_plotter_types_finds_builtin_plotters():
    discovered = viz_registry.discover_plotter_types()

    assert "state_line" in discovered
    assert "state_scatter" in discovered
    assert "series_line" in discovered


def test_list_plotter_types_sorted():
    types = viz_registry.list_plotter_types()

    assert types == sorted(types)
    assert "series_line" in types


def test_describe_plotter_type():
    info = viz_registry.describe_plotter_type("series_line")

    assert info["name"] == "series_line"
    assert info["type"] == "SeriesLinePlotter"
    assert "Line plot" in info["description"]


def test_describe_plotter_type_unknown():
    with pytest.raises(ValueError, match="Unknown plotter type"):
        viz_registry.describe_plotter_type("missing")
