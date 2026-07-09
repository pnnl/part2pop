import pytest

from part2pop.viz.base import Plotter


class _ConcretePlotter(Plotter):
    def __init__(self):
        super().__init__(type="dummy", config={})
        self.prep_calls = []
        self.render_calls = []

    def prep(self, source, **kwargs):
        self.prep_calls.append(source)
        return {"prepared": source}

    def render(self, prepared, ax, **kwargs):
        self.render_calls.append((prepared, ax, kwargs))
        return ax


def test_plot_source_calls_prep_then_render():
    plotter = _ConcretePlotter()
    source = object()
    ax = object()

    result = plotter.plot(source, ax)

    assert result is ax
    assert plotter.prep_calls == [source]
    assert plotter.render_calls == [({"prepared": source}, ax, {})]


def test_plot_prepared_skips_prep():
    plotter = _ConcretePlotter()
    prepared = {"ready": True}
    ax = object()

    result = plotter.plot(ax=ax, prepared=prepared, color="k")

    assert result is ax
    assert plotter.prep_calls == []
    assert plotter.render_calls == [(prepared, ax, {"color": "k"})]


def test_plot_prepared_method_calls_render():
    plotter = _ConcretePlotter()
    prepared = {"ready": True}
    ax = object()

    result = plotter.plot_prepared(prepared, ax)

    assert result is ax
    assert plotter.render_calls == [(prepared, ax, {})]


def test_plot_requires_source_or_prepared():
    plotter = _ConcretePlotter()

    with pytest.raises(ValueError, match="Either source or prepared plot data"):
        plotter.plot(ax=object())


def test_plot_requires_axes():
    plotter = _ConcretePlotter()

    with pytest.raises(ValueError, match="axes object"):
        plotter.plot(object())


def test_plot_accepts_population_keyword_alias():
    plotter = _ConcretePlotter()
    source = object()
    ax = object()

    plotter.plot(population=source, ax=ax)

    assert plotter.prep_calls == [source]


def test_plot_accepts_populations_keyword_alias():
    plotter = _ConcretePlotter()
    source = [object()]
    ax = object()

    plotter.plot(populations=source, ax=ax)

    assert plotter.prep_calls == [source]
