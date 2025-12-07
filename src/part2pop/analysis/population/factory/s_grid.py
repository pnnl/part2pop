from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("s_grid")
class SupersaturationGridVar(PopulationVariable):
    meta = VariableMeta(
        name="s_grid",
            axis_names=(),
        description="Supersaturation",
        units="%",
        short_label = 's',
        long_label = 'supersaturation',
        scale='log',
        # axis/grid defaults are centralized in analysis.defaults
        default_cfg={},
        aliases=('s','supersaturation','s_eval'),
    )
    def compute(self, population=None,as_dict=False):
        cfg = self.cfg
        out = np.asarray(cfg.get("s_grid", cfg.get("s_eval",[])),dtype=float)
        if as_dict:
            return {"s_grid": out}
        return out
    
def build(cfg=None):
    cfg = cfg or {}
    return SupersaturationGridVar(cfg)
