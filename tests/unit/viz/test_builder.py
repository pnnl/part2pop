import pytest

from part2pop.viz.builder import PlotBuilder, build_plotter


def test_plot_builder_requires_type():
    pb = PlotBuilder(type=None, config={})
    with pytest.raises(ValueError):
        pb.build()


def test_plot_builder_unknown_type(monkeypatch):
    monkeypatch.setattr(
        "part2pop.viz.builder.discover_plotter_types",
        lambda: {"known": lambda cfg: cfg},
    )
    with pytest.raises(ValueError):
        PlotBuilder("missing", {}).build()


def test_plot_builder_builds_using_registry(monkeypatch):
    constructed = object()

    def fake_factory(cfg):
        assert cfg == {"foo": "bar"}
        return constructed

    monkeypatch.setattr(
        "part2pop.viz.builder.discover_plotter_types",
        lambda: {"good": fake_factory},
    )

    result = build_plotter("good", {"foo": "bar"})
    assert result is constructed
