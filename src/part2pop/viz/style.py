# viz/style.py
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Mapping, Any, List, Sequence, Set, Tuple
import hashlib

# Shared defaults
DEFAULT_PALETTE = [
    "#0e5a8f","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
]
DEFAULT_LINESTYLES = ["-","--","-.",":"]
DEFAULT_MARKERS = ["o","s","^","D","v","P","X"]

def _ensure_cycle(values: Sequence[str] | str | None, fallback: Sequence[str]) -> List[str]:
    """
    Normalize cycle inputs to a list while copying mutable defaults.
    Accepts single strings, Sequences, or None (meaning -> fallback).
    """
    if values is None:
        values = fallback
    if isinstance(values, str):
        return [values]
    return list(values)

@dataclass
class GeomDefaults:
    """
    Collects style defaults + cycles for a geometry.
    palette/linestyles/markers accept either lists (multi-cycle) or a single string value.
    base_linestyle / base_marker define the one-off style when the cycle is disabled.
    """
    palette: Sequence[str] | str | None = None
    linestyles: Sequence[str] | str | None = None
    markers: Sequence[str] | str | None = None
    linewidth: float = 2.0
    markersize: float = 36.0
    alpha: float | None = None
    cmap: str = "viridis"
    base_linestyle: str | None = None
    base_marker: str | None = None
    cycle_linestyle_default: bool = False
    cycle_marker_default: bool = False

    def __post_init__(self) -> None:
        # copy/normalize lists so callers do not share references
        self.palette = _ensure_cycle(self.palette, DEFAULT_PALETTE)
        self.linestyles = _ensure_cycle(self.linestyles, DEFAULT_LINESTYLES)
        self.markers = _ensure_cycle(self.markers, DEFAULT_MARKERS)

    # how to combine when both color and something else cycle
    def combos(self, use_linestyle: bool, use_marker: bool) -> List[Tuple[str, str | None, str | None]]:
        """
        Returns [(color, linestyle, marker)] permutations for deterministic hashing.
        When a cycle is disabled, we still emit a single entry that carries the base style.
        """
        palette = list(self.palette)
        if use_linestyle:
            linestyles = list(self.linestyles)
        else:
            linestyles = [self.base_linestyle]
        if use_marker:
            markers = list(self.markers)
        else:
            markers = [self.base_marker]
        linestyles = [ls for ls in linestyles if ls is not None] or [None]
        markers = [mk for mk in markers if mk is not None] or [None]
        return [(c, ls, mk) for c, ls, mk in product(palette, linestyles, markers)]
    
@dataclass
class Theme:
    # Per-geometry defaults; extend as you add geoms
    geoms: Dict[str, GeomDefaults]

    def __init__(self, geoms: Dict[str, GeomDefaults] | None = None):
        self.geoms = geoms or {
            "line": GeomDefaults(linewidth=2.0, alpha=None, base_linestyle="-"),
            "scatter": GeomDefaults(linewidth=1.0, markersize=36.0, base_marker="o", cycle_marker_default=True),
            "bar": GeomDefaults(),
            "box": GeomDefaults(),
            "surface": GeomDefaults(),
        }

class StyleManager:
    """
    Plans per-series matplotlib kwargs given a geometry and series keys.
    Deterministic mapping: same key → same style across figures.
    """
    def __init__(self, theme: Theme | None = None, deterministic: bool = True):
        self.theme = theme or Theme()
        self.deterministic = deterministic

    def _index_for_key(self, key: str, i: int) -> int:
        if not self.deterministic:
            return i
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def plan(
        self,
        geom: str,
        series_keys: Iterable[str],
        *,
        overrides: Mapping[str, Dict[str, Any]] | None = None,
        cycle_linestyle: bool | None = None,
        cycle_marker: bool | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        if geom not in self.theme.geoms:
            raise ValueError(f"Unknown geom '{geom}'. Known: {list(self.theme.geoms)}")
        gd = self.theme.geoms[geom]
        # Sensible defaults per geom
        use_ls = bool(gd.linestyles) and (cycle_linestyle if cycle_linestyle is not None else gd.cycle_linestyle_default)
        use_mk = bool(gd.markers) and (cycle_marker if cycle_marker is not None else gd.cycle_marker_default)
        combos = gd.combos(use_ls, use_mk)
        ncombo = len(combos)

        # Allowed kwargs per matplotlib primitive (whitelist defensive approach)
        _ALLOWED_KWARGS_BY_GEOM: Dict[str, Set[str]] = {
            "line": {"color", "linestyle", "linewidth", "marker", "markersize", "alpha"},
            "scatter": {"c", "color", "cmap", "s", "marker", "alpha"},
            "bar": {"color", "alpha"},
            "box": {"color", "alpha"},
            "surface": {"cmap", "alpha"},
        }

        def _make_style_for_geom(gd: GeomDefaults, geom: str, color: str, linestyle: str | None, marker: str | None) -> Dict[str, Any]:
            style: Dict[str, Any] = {}
            if geom == "line":
                style["color"] = color
                if linestyle is not None:
                    style["linestyle"] = linestyle
                    style["linewidth"] = gd.linewidth
                if marker is not None:
                    style["marker"] = marker
                    # use markersize (Line2D expects points, not area). Use the same numeric value
                    style["markersize"] = gd.markersize
                if gd.alpha is not None:
                    style["alpha"] = gd.alpha
            elif geom == "scatter":
                # scatter supports colormap and 's' as area
                style["c"] = color
                if marker is not None:
                    style["marker"] = marker
                style["s"] = gd.markersize
                if gd.cmap:
                    style["cmap"] = gd.cmap
                if gd.alpha is not None:
                    style["alpha"] = gd.alpha
            else:
                # conservative fallback
                style["color"] = color
                if gd.alpha is not None:
                    style["alpha"] = gd.alpha
            return style

        styles: Dict[str, Dict[str, Any]] = {}
        for i, key in enumerate(series_keys):
            idx = self._index_for_key(key, i) % ncombo
            color, linestyle, marker = combos[idx]
            base = _make_style_for_geom(gd, geom, color, linestyle, marker)
            # apply overrides if any (caller may pass geom-appropriate keys)
            if overrides and key in overrides:
                base.update(overrides[key])
            # whitelist/filter unknown kwargs
            allowed: Set[str] = _ALLOWED_KWARGS_BY_GEOM.get(geom, set(base.keys()))
            styles[key] = {k: v for k, v in base.items() if k in allowed}
        return styles
