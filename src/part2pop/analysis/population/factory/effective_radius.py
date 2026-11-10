from __future__ import annotations

import numpy as np

from ..base import PopulationVariable, VariableMeta
from .registry import register_variable


@register_variable("effective_radius")
class EffectiveRadiusVar(PopulationVariable):
    meta = VariableMeta(
        name="effective_radius",
        axis_names=(),
        description="Area-weighted effective radius",
        units="m",
        scale="linear",
        long_label="Effective radius",
        short_label="$r_{eff}$",
    )

    def compute(self, population, as_dict=False):
        # Wet radii at the current water content; no humidity equilibration here.
        radii = np.asarray(
            [population.get_particle(pid).get_Dwet() / 2.0 for pid in population.ids]
        )
        num_concs = np.asarray(population.num_concs, dtype=float)
        area_moment = np.sum(num_concs * radii**2)
        if area_moment == 0:
            value = np.nan
        else:
            value = float(np.sum(num_concs * radii**3) / area_moment)
        if as_dict:
            return {"effective_radius": value}
        return value


def build(cfg=None):
    return EffectiveRadiusVar(cfg or {})
