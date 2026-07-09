import numpy as np

from .registry import register
from ..base import StatePlotter
from ...analysis import build_variable


def _as_1d(value):
    if value is None:
        return None

    arr = np.asarray(value)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return arr.reshape(1)

    if arr.ndim != 1:
        raise ValueError(
            f"state_line needs 1-D data after removing singleton dimensions; "
            f"got shape {np.asarray(value).shape}."
        )

    return arr


@register("state_line")
class StateLinePlotter(StatePlotter):
    def __init__(self, config: dict):
        self.type = "state_line"
        self.config = config
        self.varname = config.get("varname")
        self.var_cfg = dict(config.get("var_cfg", {}))
        if not self.varname:
            raise ValueError("StateLinePlotter requires 'varname' in config.")
        # normalize synonyms
        if "wvls" in self.var_cfg and "wvl_grid" not in self.var_cfg:
            self.var_cfg["wvl_grid"] = self.var_cfg.pop("wvls")
        
    def _fmt_label(self, long_label, units):
        units = (units or "").strip()
        return f"{long_label} [{units}]" if units else long_label
    
    def prep(self, population):
        # choose x-axis variable (simplified)
        if self.varname in ("Nccn", "frac_ccn"):
            xvar = build_variable("s_grid", scope="population", var_cfg=self.var_cfg)
        elif self.varname == "avg_Jhet":
            xvar = build_variable("T_grid", scope="population", var_cfg=self.var_cfg)
        elif self.varname in ("frozen_frac", "unfrozen_frac"):
            xvar=build_variable("time_grid", "population",  self.var_cfg)
        elif self.varname in ("b_abs","b_scat","b_ext"):
            has_w = len(self.var_cfg.get("wvl_grid", [])) > 1
            has_rh = len(self.var_cfg.get("rh_grid", [])) > 1
            if has_w and has_rh:
                raise ValueError("state_line needs one varying axis (wavelength or RH).")
            elif has_w:
                xvar = build_variable(name="wvl_grid", scope="population", var_cfg=self.var_cfg)
            elif has_rh:
                xvar = build_variable(name="rh_grid", scope="population", var_cfg=self.var_cfg)
            else:
                raise ValueError(f"Variable {self.varname} has single wavelength and single RH value; cannot plot state line.")
            #xvar = build_variable("wvl_grid" if has_w else "rh_grid", "population", self.var_cfg)
        elif self.varname == "dNdlnD":
            xvar = build_variable("diam_grid","population",  self.var_cfg)
        elif self.varname == "INSA_distribution":
            xvar = build_variable(name="INSA_grid", scope="population", var_cfg=self.var_cfg)
        else:
            raise ValueError(f"State line does not support '{self.varname}'.")

        yvar = build_variable(name=self.varname, scope="population", var_cfg=self.var_cfg)
        x = _as_1d(xvar.compute(population))
        y = _as_1d(yvar.compute(population))

        if x is not None and len(x) != len(y):
            raise ValueError(f"x and y must be same length, got {len(x)} vs {len(y)}.")

        return {
            "x": x, "y": y,
            "xlabel": self._fmt_label(xvar.meta.long_label, getattr(xvar.meta, "units", "")),
            "ylabel": self._fmt_label(yvar.meta.long_label, getattr(yvar.meta, "units", "")),
            "xscale": xvar.meta.scale, "yscale": yvar.meta.scale,
        }

    def render(self, prepared, ax, add_ylabel=True, add_xlabel=True, **kwargs):
        style = {**self.config.get("style", {}), **kwargs}
        x = _as_1d(prepared["x"])
        y = _as_1d(prepared["y"])
        if x is None:
            ax.plot(y, **style)
        else:
            ax.plot(x, y, **style)
        if add_xlabel:
            ax.set_xlabel(prepared["xlabel"])
        
        if add_ylabel:
            ax.set_ylabel(prepared["ylabel"])
        
        ax.set_xscale(prepared["xscale"]); ax.set_yscale(prepared["yscale"])
        
        # FIXME: should this be in here or elsewhere?
        if x is not None and len(x) > 1:
            ax.set_xlim(np.nanmin(x), np.nanmax(x))
        return ax

def build(cfg):
    return StateLinePlotter(cfg)
