# viz/style.py
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple, Iterable, Mapping, Any, Set
import hashlib

# Shared defaults
DEFAULT_PALETTE = [
    "#0e5a8f","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
]
DEFAULT_LINESTYLES = ["-","--","-.",":"]
DEFAULT_MARKERS = ["o","s","^","D","v","P","X"]

@dataclass
class GeomDefaults:
    # Discrete cycles
    palette: List[str] = field(default_factory=lambda: DEFAULT_PALETTE.copy())
    linestyles: List[str] = field(default_factory=lambda: DEFAULT_LINESTYLES.copy())
    markers: List[str] = field(default_factory=lambda: DEFAULT_MARKERS.copy())
    # Scalar defaults
    linewidth: float = 2.0
    markersize: float = 36.0  # matplotlib 's' is area in points^2
    alpha: float | None = None
    # Continuous mappings
    cmap: str = "viridis"  # for scatter/surface color mapping
    
    # how to combine when both color and something else cycle
    def combos(self, use_linestyle: bool, use_marker: bool) -> List[Tuple[str, str | None, str | None]]:
        # (color, linestyle, marker)
        if use_linestyle and use_marker:
            return [(c, ls, mk) for c, ls, mk in product(self.palette, self.linestyles, self.markers)]
        if use_linestyle:
            return [(c, ls, None) for c, ls in product(self.palette, self.linestyles)]
        if use_marker:
            return [(c, None, mk) for c, mk in product(self.palette, self.markers)]
        return [(c, None, None) for c in self.palette]

@dataclass
class Theme:
    # Per-geometry defaults; extend as you add geoms
    geoms: Dict[str, GeomDefaults] = field(default_factory=lambda: {
        "line": GeomDefaults(linewidth=2.0, alpha=None, linestyles='-'),
        "scatter": GeomDefaults(linewidth=1.0, markersize=36.0),
        "bar": GeomDefaults(),
        "box": GeomDefaults(),
        "surface": GeomDefaults(),
    })

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
        use_ls = gd.linestyles and (cycle_linestyle if cycle_linestyle is not None else geom == "line")
        use_mk = gd.markers and (cycle_marker if cycle_marker is not None else geom == "scatter")
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
