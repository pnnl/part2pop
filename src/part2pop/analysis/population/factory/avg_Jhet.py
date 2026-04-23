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
        default_cfg={},
        aliases=("J_het"),
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        morphology = cfg.get("morphology", "homogeneous")
        species_modifications = cfg.get("species_modifications", None)
        T_grid = cfg.get("T_grid", None)
        T_units = cfg.get("T_units", None)
        RH = cfg.get("RH", 0.85)
        if T_grid is None and T_units is None:
            T_grid = np.linspace(233.15, 273.15, 50)
            T_units = "K"
        if T_units=="K" and np.min(T_grid)<=0:
            raise ValueError(f"One or more temperatures in T_grid is < 0.0 K when plotting avg_Jhet.")
        elif T_units not in ("K", "C"):
            raise ValueError(f"Unrecognized temperature units: {T_units}.")       
        
        # equilibrate population to RH
        T_grid = np.asarray(T_grid)
        if T_units=="C":
            T_grid = T_grid+273.15
        population._equilibrate_h2o(RH, T_grid[0])

        # override the underlying population species_modifications if one is supplied
        if species_modifications:
            population.species_modifications = species_modifications
        else:
            species_modifications = population.species_modifications

        # make freezing population
        freezing_config={"morphology": morphology,
                         "species_modifications": species_modifications}        
        freezing_pop = build_freezing_population(population, freezing_config)
        arr = np.zeros(len(T_grid), dtype=float)
        for ii, T in enumerate(T_grid):
            arr[ii] = freezing_pop.get_avg_Jhet(T, freezing_config)
        if as_dict:
            return {"T_grid": np.asarray(T_grid), "T_units": T_grid, "avg_Jhet": arr}
        return arr


def build(cfg=None):
    cfg = cfg or {}
    return avgJhetVar(cfg)
