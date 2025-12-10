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
    def compute(self, population=None,as_dict=False):
        cfg = self.cfg
        units = cfg.get("T_units", "K")
        if units=="C":        
            out = np.asarray(cfg.get("T_grid", []))+273.15
        elif units=="K":
            out = np.asarray(cfg.get("T_grid", []))
        else:
            raise ValueError(f"Unknown temperature unit: '{units}'.")
        if as_dict:
            return {"T_grid": out}
        return out
    
def build(cfg=None):
    cfg = cfg or {}
    return TemperatureGridVar(cfg)
