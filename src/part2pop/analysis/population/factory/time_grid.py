from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
#from part2pop.optics.builder import build_optical_population

@register_variable("time_grid")
class TimeGridVar(PopulationVariable):
    meta = VariableMeta(
        name="time_grid",
            axis_names=(),
        description="Time",
        units="s",
        short_label = "t",
        long_label = "time",
        scale="linear",
        # axis/grid defaults are centralized in analysis.defaults
        default_cfg={},
        aliases=(),
    )
    def compute(self, population=None,as_dict=False):
        cfg = self.cfg
        t_min = cfg.get("t_min", 0.0)
        t_max = cfg.get("t_max", 360.0)
        dt = cfg.get("dt", 0.5)
        time = np.arange(t_min, t_max+dt, dt)
        if population:
            time = np.repeat(time[:, None], len(population.T_grid), axis=1)
        return time
    
def build(cfg=None):
    cfg = cfg or {}
    return TimeGridVar(cfg)
