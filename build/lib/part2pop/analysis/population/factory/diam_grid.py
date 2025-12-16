from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("diam_grid")
class DiamGridVar(PopulationVariable):
    meta = VariableMeta(
        name="diam_grid",
        axis_names=(),
        description="Particle dry diameter grid",
        units="m",
        scale='log',
        long_label = 'dry diameter',
        short_label = 'D',
    # axis/grid defaults are centralized in analysis.defaults
    default_cfg={},
        aliases=("D"),
    )
    def compute(self, population,as_dict=False):
        cfg = self.cfg
        vals = cfg.get("diam_grid")
        if as_dict:
            return {"diam_grid": np.asarray(vals)}
        else:
            return np.asarray(vals)

def build(cfg=None):
    cfg = cfg or {}
    return DiamGridVar(cfg)
