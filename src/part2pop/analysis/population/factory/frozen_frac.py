from __future__ import annotations
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_population
import numpy as np

@register_variable("frozen_frac")
class FrozenFraction(PopulationVariable):
    meta = VariableMeta(
        name="frozen_frac",
        description='Fraction of frozen particles.',
        units = r'',
        axis_names=("T_grid"),
        default_cfg={},
        aliases = ('F_frz',),
        scale = 'linear',
        short_label = 'frozen_frac',
        long_label = 'frozen fraction',
    )
    
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        units = cfg.get("T_units", "K")
        freezing_cfg = {
            "T_grid": list(cfg["T_grid"]),
            "morphology": cfg.get("morphology", "homogeneous"),
            "species_modifications": cfg.get("species_modifications", {}),
            "T_units": units
        }
        freezing_pop = build_freezing_population(population, freezing_cfg)     
        arr = freezing_pop.get_frozen_fraction(cfg["cooling_rate"])
        if as_dict:
            return {"T_grid": np.asarray(cfg["T_grid"]), "cooling_rate": cfg["cooling_rate"],
                    "T_units": cfg.get("T_units", "K"), "nucleating_sites": arr,}
        return arr
    
    
def build(cfg=None):
    cfg = cfg or {}
    return FrozenFraction(cfg)
