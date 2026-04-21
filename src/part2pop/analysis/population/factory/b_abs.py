from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

@register_variable("b_abs")
class AbsCoeff(PopulationVariable):
    meta = VariableMeta(
            name="b_abs",
            axis_names=("rh_grid", "wvls"),
        description="Absorption coefficient",
        units="m$^{-1}$",
        short_label="$b_{\mathrm{abs}}$",
        long_label="absorption coefficient",
        scale='linear',
        aliases=("total_abs",),
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
        arr = optical_pop.get_optical_coeff("b_abs", rh=None, wvl=None)
        if as_dict:
            return {"rh_grid": np.asarray(cfg["rh_grid"]), "wvl_grid": np.asarray(ocfg["wvl_grid"]), "b_abs": arr}
        return arr


def build(cfg=None):
    """
    Canonical factory entry point for analysis variables.
    Mirrors b_scat/b_ext API so runners can uniformly call module.build(cfg).
    """
    cfg = cfg or {}
    return AbsCoeff(cfg)
