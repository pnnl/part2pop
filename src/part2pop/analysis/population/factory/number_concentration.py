from __future__ import annotations

from ..base import PopulationVariable, VariableMeta
from .registry import register_variable


@register_variable("number_concentration")
class NumberConcentrationVar(PopulationVariable):
    meta = VariableMeta(
        name="number_concentration",
        axis_names=(),
        description="Total aerosol number concentration",
        units="m$^{-3}$",
        scale="linear",
        long_label="Aerosol number concentration",
        short_label="$N$",
        aliases=("Ntot", "N_total"),
    )

    def compute(self, population, as_dict=False):
        value = float(population.get_Ntot())
        if as_dict:
            return {"number_concentration": value}
        return value


def build(cfg=None):
    return NumberConcentrationVar(cfg or {})