from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("rh_grid")
class RHGridVar(PopulationVariable):
    meta = VariableMeta(
        name="rh_grid",
            axis_names=(),
        description="Relative humidity",
        units="", # often converted to percent
        # todo: need rescale function -- fraction to percent
        scale='linear',
        long_label = 'relative humidity',
        short_label = 'RH',
        # axis/grid defaults are centralized in analysis.defaults
        default_cfg={},
        aliases=(),
    )
    def compute(self, population,as_dict=False):
        cfg = self.cfg
        vals = cfg.get("rh_grid", np.array([0.0]))
        if as_dict:
            return {"rh_grid": np.asarray(vals)}
        else:
            return np.asarray(vals)

def build(cfg=None):
    cfg = cfg or {}
    return RHGridVar(cfg)
