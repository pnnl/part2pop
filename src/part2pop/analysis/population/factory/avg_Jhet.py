from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_population

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
        default_cfg={
            "morphology": "homogeneous",
            "species_modifications": {},
            "T_grid": [298.15],
        },
        aliases=("J_het"),
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        freezing_cfg = {
            "T_grid": list(cfg["T_grid"]),
            "morphology": cfg.get("morphology", "homogeneous"),
            "species_modifications": cfg.get("species_modifications", {}),
            "T_units": cfg.get("T_units", "K")
        }
        freezing_pop = build_freezing_population(population, freezing_cfg)        
        arr = freezing_pop.get_avg_Jhet()
        if as_dict:
            return {"T_grid": np.asarray(cfg["T_grid"]), "T_units": cfg.get("T_units", "K"), "avg_Jhet": arr}
        return arr


def build(cfg=None):
    cfg = cfg or {}
    return avgJhetVar(cfg)
