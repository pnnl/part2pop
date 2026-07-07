from __future__ import annotations

from dataclasses import replace

from ..base import PopulationVariable, VariableMeta
from .registry import register_variable


@register_variable("mass_concentration")
class MassConcentrationVar(PopulationVariable):
    meta = VariableMeta(
        name="mass_concentration",
        axis_names=(),
        description="Total aerosol mass concentration",
        units="kg m$^{-3}$",
        scale="linear",
        long_label="Aerosol mass concentration",
        short_label="$M$",
        default_cfg={"dry": False},
        aliases=("Mtot", "M_total"),
    )

    def compute(self, population, as_dict=False):
        if self.cfg.get("dry", False):
            value = float(population.get_tot_dry_mass())
        else:
            value = float(population.get_tot_mass())

        if as_dict:
            return {"mass_concentration": value}
        return value


def build(cfg=None):
    var = MassConcentrationVar(cfg or {})

    if var.cfg.get("dry", False):
        var.meta = replace(
            var.meta,
            long_label="Dry aerosol mass concentration",
            short_label="$M_{dry}$",
        )

    return var