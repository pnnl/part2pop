from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
#from part2pop.optics.builder import build_optical_population

@register_variable("T_grid")
class TemperatureGridVar(PopulationVariable):
    meta = VariableMeta(
        name="T_grid",
            axis_names=(),
        description="Temperature",
        units="K",
        short_label = 'T',
        long_label = 'temperature',
        scale='linear',
        # axis/grid defaults are centralized in analysis.defaults
        default_cfg={},
        aliases=('T','temp','temperature', 'T_eval'),
    )
    def compute(self, population=None, as_dict=False):
        cfg = self.cfg
        units = cfg.get("T_units", "K")
        T_grid = cfg.get("T_grid", np.linspace(233.15, 273.15, 50))       
        out = np.asarray(T_grid)
        if units not in ("K","C"):
            raise ValueError(f"Unknown temperature unit: '{units}'.")
        if units=="K" and np.min(T_grid)<=0:
            raise ValueError(f"One or more values of T_grid is <= 0 K.")
        if as_dict:
            return {"T_grid": out}
        return out
    
def build(cfg=None):
    var = TemperatureGridVar(cfg or {})
    var.meta.units = "K"
    if var.cfg.get("T_units")=="C":
        var.meta.units = "C"
    return var
