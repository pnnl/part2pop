from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("INSA_grid")
class InsaGridVar(PopulationVariable):
    meta = VariableMeta(
        name="INSA_grid",
        axis_names=(),
        description="Particle ice nucleating surface area grid",
        units="m$^2$",
        scale="log",
        long_label = "ice nucelating surface area",
        short_label = "INSA",
    # axis/grid defaults are centralized in analysis.defaults
    default_cfg={},
    #aliases=("insa_grid"),
    )
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        vals = cfg.get("insa_grid")
        if as_dict:
            return {"insa_grid": np.asarray(vals)}
        else:
            return np.asarray(vals)

def build(cfg=None):
    cfg = cfg or {}
    return InsaGridVar(cfg)
