import pytest

from part2pop.viz.builder import PlotBuilder, build_plotter
from part2pop.viz.factory.series_line import SeriesLinePlotter


def test_plot_builder_requires_type():
    pb = PlotBuilder(type=None, config={})
    with pytest.raises(ValueError):
        pb.build()


def test_plot_builder_unknown_type():
    with pytest.raises(ValueError, match="Unknown plotter type"):
        PlotBuilder("missing", {}).build()


def test_plot_builder_builds_registered_plotter():
    result = build_plotter("series_line", {"varname": "Nccn"})

    assert isinstance(result, SeriesLinePlotter)
    assert result.config == {"varname": "Nccn"}
