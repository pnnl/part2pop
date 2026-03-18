from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_particle

@register_variable("avg_Jhet")
class avgJhetVar(PopulationVariable):
    meta = VariableMeta(
        name="avg_Jhet",
        axis_names=("T_grid"),
        description="Average heterogeneous ice nucleation rate.",
        units="m$^{-2}$s$^{-1}$",
        short_label="$J_{het}$",
        long_label="ice nucleation rate",
        scale='log',
        # axis/grid defaults are centralized in analysis.defaults; keep other defaults
        default_cfg={},
        aliases=("J_het"),
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        T_units = cfg.get("T_units", "K")
        if T_units=="K" and np.min(population.T_grid)<=0:
            raise ValueError(f"One or more temperatures in T_grid is < 0.0 K when plotting avg_Jhet.")
        elif T_units not in ("K", "C"):
            raise ValueError(f"Unrecognized temperature units: f{T_units}.")       
        
        arr = population.get_avg_Jhet()
        
        if T_units == "C":
            T_grid = population.T_grid-273.15
        else:
            T_grid = population.T_grid
        if as_dict:
            return {"T_grid": np.asarray(T_grid), "T_units": T_grid, "avg_Jhet": arr}
        return arr


def build(cfg=None):
    cfg = cfg or {}
    return avgJhetVar(cfg)
