from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("b_ext")
class ExtinctionCoeff(PopulationVariable):
    meta = VariableMeta(
        name="b_ext",
        axis_names=("rh_grid", "wvls"),
        description="Extinction coefficient",
        units="m$^{-1}$",
        short_label="$b_{\mathrm{text}}$",
        long_label="extinction coefficient",
        scale='linear',
        # axis/grid defaults are centralized in analysis.defaults; keep other defaults
        default_cfg={
            "morphology": "core-shell",
            "species_modifications": {},
            "T": 298.15,
        },
        aliases=("total_ext","extinction_coeff","ext_coeff","extinction_coefficient",),
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        morph = cfg["morphology"]
        if morph == "core-shell":
            morph = "core_shell"
        ocfg = {
            "rh_grid": cfg["rh_grid"],
            "wvl_grid": cfg["wvl_grid"],
            "type": morph,
            "temp": cfg["T"],
            "species_modifications": cfg.get("species_modifications", {}),
        }
        optical_pop = build_optical_population(population, ocfg)
        arr = optical_pop.get_optical_coeff("b_ext", rh=None, wvl=None)
        if as_dict:
            return {"rh_grid": np.asarray(cfg["rh_grid"]), "wvl_grid": np.asarray(cfg["wvl_grid"]), "b_ext": arr}
        else:
            return arr

def build(cfg=None):
    cfg = cfg or {}
    return ExtinctionCoeff(cfg)
