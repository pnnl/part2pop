# viz/factory/state_scatter.py
from .registry import register
from ..base import Plotter
from ...analysis import build_variable


# FIXME: in progress
@register("state_scatter")
class StateScatterPlotter(Plotter):
    """
    Scatter plot using variables:
      required: xvar, yvar
      optional: cvar (color), svar (size)
    Config:
      {
        "xvar": "diam_grid",
        "yvar": "b_ext",
        "cvar": "Nccn",         # optional → maps to 'c' + cmap
        "svar": "dNdlnD",       # optional → maps to 's' (marker size)
        "var_cfg": {...},       # shared cfg for variables (simple case)
        "style": {...},         # pre-planned style from StyleManager(plan("scatter", ...))
        "colorbar": True,       # show colorbar if cvar provided
        "clabel": "Nccn [cm⁻³]" # optional colorbar label; else from meta
      }
    """
    def __init__(self, config: dict):
        self.type = "state_scatter"
        self.config = config
        # make config for each variable? (see parci processes)
        self.var_cfg = dict(config.get("var_cfg", {}))
        self.xname = config.get("xvar")
        self.yname = config.get("yvar")
        self.cname = config.get("cvar",None)
        self.sname = config.get("svar",None)
        if not (self.xname and self.yname):
            raise ValueError("StateScatterPlotter requires 'xvar' and 'yvar' in config.")
        
    def _fmt_label(self, long_label, units):
        units = (units or "").strip()
        return f"{long_label} [{units}]" if units else long_label

    def prep(self, population):
        # fixme: make these particle variables -- different function?
        xvar = build_variable(self.xname, 'particle', self.var_cfg)
        yvar = build_variable(self.yname, 'particle', self.var_cfg)
        x = xvar.compute_all(population)
        y = yvar.compute_all(population)

        if len(x) != len(y):
            raise ValueError(f"x and y must be same length, got {len(x)} vs {len(y)}.")

        # optional color & size channels
        c = s = None
        clabel = None
        if self.cname:
            cvar = build_variable(self.cname, self.var_cfg)
            c = cvar.compute(population)
            if len(c) != len(x):
                raise ValueError("cvar length must match x/y length.")
            clabel = self._fmt_label(cvar.meta.long_label, getattr(cvar.meta, "units", ""))

        if self.sname:
            svar = build_variable(self.sname, self.var_cfg)
            s = svar.compute(population)
            if len(s) != len(x):
                raise ValueError("svar length must match x/y length.")

        return {
            "x": x, "y": y, "c": c, "s": s,
            "xlabel": self._fmt_label(xvar.meta.long_label, getattr(xvar.meta, "units", "")),
            "ylabel": self._fmt_label(yvar.meta.long_label, getattr(yvar.meta, "units", "")),
            "clabel": self.config.get("clabel", clabel),
            "xscale": xvar.meta.scale, "yscale": yvar.meta.scale
        }

    def plot(self, population, ax, **kwargs):
        pd = self.prep(population)        
        style = {**self.config.get("style", {}), **kwargs}
        # If c provided, prefer continuous mapping (use cmap in style/theme)
        scatter_kwargs = dict(style)
        if pd["c"] is not None:
            scatter_kwargs["c"] = pd["c"]
            # ensure a cmap exists (StyleManager sets it)
            scatter_kwargs.setdefault("cmap", style.get("cmap", "viridis"))
        if pd["s"] is not None:
            scatter_kwargs["s"] = pd["s"]

        h = ax.scatter(pd["x"], pd["y"], **scatter_kwargs)
        ax.set_xlabel(pd["xlabel"]); ax.set_ylabel(pd["ylabel"])
        ax.set_xscale(pd["xscale"]); ax.set_yscale(pd["yscale"])

        # optional colorbar
        if pd["c"] is not None and self.config.get("colorbar", True):
            cb = ax.figure.colorbar(h, ax=ax)
            if pd.get("clabel"):
                cb.set_label(pd["clabel"])

        return ax
    
def build(cfg):
    return StateScatterPlotter(cfg)