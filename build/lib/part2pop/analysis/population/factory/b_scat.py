from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("b_scat")
class BScatVar(PopulationVariable):
    meta = VariableMeta(
        name="b_scat",
        axis_names=("rh_grid", "wvls"),
        description="Scattering coefficient",
        units="m$^{-1}$",
        short_label="$b_{scat}$",
        long_label="scattering coefficient",
        scale='linear',
        # axis/grid defaults are centralized in analysis.defaults; keep other defaults
        default_cfg={
            "morphology": "core-shell",
            "species_modifications": {},
            "T": 298.15,
        },
        aliases=("total_scat",),
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        morph = cfg["morphology"]
        if morph == "core-shell":
            morph = "core_shell"
        ocfg = {
            "rh_grid": list(cfg["rh_grid"]),
            "wvl_grid": list(cfg.get("wvl_grid", cfg.get("wvls", []))),
            "type": morph,
            "temp": cfg["T"],
            "species_modifications": cfg.get("species_modifications", {}),
        }
        optical_pop = build_optical_population(population, ocfg)
        # print(optical_pop.get_particle(1).get_refractive_index(550))
        arr = optical_pop.get_optical_coeff("b_scat", rh=None, wvl=None)
        if as_dict:
            return {"rh_grid": np.asarray(cfg["rh_grid"]), "wvl_grid": np.asarray(ocfg["wvl_grid"]), "b_scat": arr}
        return arr


def build(cfg=None):
    cfg = cfg or {}
    return BScatVar(cfg)
