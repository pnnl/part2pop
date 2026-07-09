from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Plotter:
    """Base class for plotters.

    Plotters transform caller-provided source data into prepared plot data,
    then render prepared plot data onto a matplotlib axes.
    """
    type: str
    config: dict

    source_kind: ClassVar[str] = "generic"

    def prep(self, source, **kwargs):
        raise NotImplementedError

    def render(self, prepared, ax, **kwargs):
        raise NotImplementedError

    def plot(self, source=None, ax=None, *, prepared=None, **kwargs):
        if source is None and "population" in kwargs:
            source = kwargs.pop("population")
        if source is None and "populations" in kwargs:
            source = kwargs.pop("populations")

        if ax is None:
            raise ValueError("An axes object must be provided.")

        if prepared is None:
            if source is None:
                raise ValueError("Either source or prepared plot data must be provided.")
            prepared = self.prep(source)

        return self.render(prepared, ax, **kwargs)

    def plot_prepared(self, prepared, ax, **kwargs):
        return self.render(prepared, ax, **kwargs)


class StatePlotter(Plotter):
    """Plotter whose raw source is one ParticlePopulation."""

    source_kind: ClassVar[str] = "population"


class SeriesPlotter(Plotter):
    """Plotter whose raw source is a sequence/table of population records."""

    source_kind: ClassVar[str] = "population_series"
