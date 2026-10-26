from __future__ import annotations
import numpy as np
from part2pop.analysis._humidity import validate_equilibration_rh
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_population
from part2pop.analysis._freezing import resolve_freezing_config

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
        """Evaluate a finite 1-D temperature grid above absolute zero.

        Validate and convert the entire grid before equilibrating water. This
        input check does not define an empirical nucleation-fit validity range.
        """
        cfg = self.cfg
        morphology = cfg.get("morphology", "homogeneous")
        species_modifications = resolve_freezing_config(population, cfg)["species_modifications"]
        T_grid = cfg.get("T_grid", None)
        T_units = cfg.get("T_units", None)
        RH = validate_equilibration_rh(cfg.get("RH", 0.85))
        if T_grid is None and T_units is None:
            T_grid = np.linspace(233.15, 273.15, 50)
            T_units = "K"
        if T_units not in ("K", "C"):
            raise ValueError(f"Unrecognized temperature units: {T_units}.")
        try:
            T_grid = np.asarray(T_grid, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("T_grid must be a nonempty one-dimensional numeric grid.") from exc
        if T_grid.ndim != 1 or T_grid.size == 0:
            raise ValueError("T_grid must be a nonempty one-dimensional numeric grid.")
        if T_units == "C":
            T_grid = T_grid + 273.15
        if not np.all(np.isfinite(T_grid)) or np.any(T_grid <= 0):
            raise ValueError("T_grid must contain only finite temperatures > 0 K.")

        # Equilibrate only after every temperature has passed validation.
        population._equilibrate_h2o(RH, T_grid[0])

        # Species overrides apply to this calculation, not population defaults.

        # make freezing population
        freezing_config={"morphology": morphology,
                         "species_modifications": species_modifications}        
        freezing_pop = build_freezing_population(population, freezing_config)
        arr = np.zeros(len(T_grid), dtype=float)
        for ii, T in enumerate(T_grid):
            arr[ii] = freezing_pop.get_avg_Jhet(T, freezing_config)
        if as_dict:
            return {"T_grid": np.asarray(T_grid), "T_units": "K", "avg_Jhet": arr}
        return arr


def build(cfg=None):
    cfg = cfg or {}
    return avgJhetVar(cfg)
