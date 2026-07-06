from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from .registry import register
from ..base import SeriesPlotter
from ..style import StyleManager
from ...analysis import build_variable


def _iter_rows(data):
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


def _row_get(row, key):
    try:
        return row[key]
    except (KeyError, TypeError, IndexError) as exc:
        raise KeyError(f"Missing required column/key '{key}' for series_line plot data.") from exc


def _row_to_dict(row):
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _scalar_y(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    raise ValueError(
        "series_line requires scalar y values; use state_line for vector "
        "diagnostics or configure an explicit reducer in the future."
    )


def _fmt_label(long_label, units):
    units = (units or "").strip()
    return f"{long_label} [{units}]" if units else long_label


@register("series_line")
class SeriesLinePlotter(SeriesPlotter):
    """Line plot for scalar diagnostics across multiple population records."""

    def __init__(self, config: dict):
        self.type = "series_line"
        self.config = config
        self.varname = config.get("varname")
        self.var_cfg = dict(config.get("var_cfg", {}))
        self.x_key = config.get("x", "x")
        self.population_key = config.get("population", "population")
        self.series_key = config.get("series")
        self.group_key = config.get("group")
        self.y_key = config.get("y", "y")
        self.xlabel = config.get("xlabel")
        self.ylabel = config.get("ylabel")
        self.xscale = config.get("xscale", "linear")
        self.yscale = config.get("yscale")
        self.legend = config.get("legend", True)
        self.sort = config.get("sort", True)

    def prep(self, records):
        if not self.varname:
            raise ValueError("SeriesLinePlotter requires 'varname' in config when preparing records.")

        yvar = build_variable(name=self.varname, scope="population", var_cfg=self.var_cfg)
        rows = []
        for record in _iter_rows(records):
            record = _row_to_dict(record)
            population = _row_get(record, self.population_key)

            row = {key: value for key, value in record.items() if key != self.population_key}
            row[self.x_key] = _row_get(record, self.x_key)
            row[self.y_key] = _scalar_y(yvar.compute(population))
            rows.append(row)

        ymeta = getattr(yvar, "meta", None)
        ylabel = self.ylabel
        if ylabel is None and ymeta is not None:
            ylabel = _fmt_label(getattr(ymeta, "long_label", self.varname), getattr(ymeta, "units", ""))

        return {
            "data": rows,
            "x": self.x_key,
            "y": self.y_key,
            "series": self.series_key,
            "group": self.group_key,
            "xlabel": self.xlabel or self.x_key,
            "ylabel": ylabel or self.varname,
            "xscale": self.xscale,
            "yscale": self.yscale or getattr(ymeta, "scale", "linear"),
            "sort": self.sort,
        }

    def render(self, prepared, ax, **kwargs):
        for required in ("data", "x", "y"):
            if required not in prepared:
                raise ValueError(f"series_line prepared data requires '{required}'.")

        rows = _iter_rows(prepared["data"])
        x_col = prepared["x"]
        y_col = prepared["y"]
        series_col = prepared.get("series", self.series_key)
        group_col = prepared.get("group", self.group_key)
        sort_rows = prepared.get("sort", self.sort)

        lines = OrderedDict()
        for row in rows:
            series_value = _row_get(row, series_col) if series_col else None
            group_value = _row_get(row, group_col) if group_col else None
            key = (series_value, group_value)
            lines.setdefault(key, []).append(row)

        line_labels = [self._line_label(key, series_col, group_col) for key in lines]
        style_keys = [str(label if label is not None else "__all__") for label in line_labels]
        styles = StyleManager().plan(
            "line",
            style_keys,
            cycle_linestyle=bool(series_col and group_col),
        )

        any_label = False
        for (key, line_rows), label, style_key in zip(lines.items(), line_labels, style_keys):
            if sort_rows:
                line_rows = sorted(line_rows, key=lambda row: _row_get(row, x_col))

            line_style = dict(styles.get(style_key, {}))
            line_style.update(self.config.get("style", {}))
            line_style.update(kwargs)
            if label is not None:
                line_style["label"] = label
                any_label = True

            x = [_row_get(row, x_col) for row in line_rows]
            y = [_row_get(row, y_col) for row in line_rows]
            ax.plot(x, y, **line_style)

        ax.set_xlabel(prepared.get("xlabel") or self.xlabel or x_col)
        ax.set_ylabel(prepared.get("ylabel") or self.ylabel or y_col)
        ax.set_xscale(prepared.get("xscale") or self.xscale or "linear")
        ax.set_yscale(prepared.get("yscale") or self.yscale or "linear")

        if self.legend and prepared.get("legend", True) and any_label:
            ax.legend()

        return ax

    def _line_label(self, key, series_col, group_col):
        series_value, group_value = key
        if series_col and group_col:
            return f"{series_value} / {group_value}"
        if series_col:
            return str(series_value)
        if group_col:
            return str(group_value)
        return None


def build(cfg):
    return SeriesLinePlotter(cfg)
