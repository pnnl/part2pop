from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..base import PopulationVariable, VariableMeta
from .registry import register_variable


def _diameters_and_weights(population, *, wetsize: bool = True):
    """Return positive particle diameters and number-concentration weights."""
    diameter_getter = "get_Dwet" if wetsize else "get_Ddry"

    diameters = np.asarray(
        [
            getattr(population.get_particle(pid), diameter_getter)()
            for pid in population.ids
        ],
        dtype=float,
    )
    weights = np.asarray(population.num_concs, dtype=float)

    if diameters.shape[0] != weights.shape[0]:
        raise ValueError(
            "Diameter and number concentration arrays must have the same length."
        )

    mask = (
        np.isfinite(diameters)
        & np.isfinite(weights)
        & (diameters > 0.0)
        & (weights > 0.0)
    )

    return diameters[mask], weights[mask]


@register_variable("geom_std_dev")
class GeomStdDevVar(PopulationVariable):
    meta = VariableMeta(
        name="geom_std_dev",
        axis_names=(),
        description="Number-weighted geometric standard deviation of particle diameter",
        units="",
        scale="linear",
        long_label="Geometric standard deviation",
        short_label="$\\sigma_g$",
        default_cfg={"wetsize": True},
        aliases=(
            "geometric_standard_deviation",
            "geometric_std_dev",
            "GSD",
            "sigma_g",
        ),
    )

    def compute(self, population, as_dict: bool = False):
        diameters, weights = _diameters_and_weights(
            population,
            wetsize=bool(self.cfg.get("wetsize", True)),
        )

        if diameters.size == 0:
            value = np.nan
        else:
            log_diameters = np.log(diameters)
            mean_log_diameter = np.sum(weights * log_diameters) / np.sum(weights)
            var_log_diameter = (
                np.sum(weights * (log_diameters - mean_log_diameter) ** 2)
                / np.sum(weights)
            )
            value = float(np.exp(np.sqrt(var_log_diameter)))

        if as_dict:
            return {"geom_std_dev": value}

        return value


def build(cfg=None):
    cfg = cfg or {}
    var = GeomStdDevVar(cfg)

    if cfg.get("wetsize", True):
        var.meta = replace(
            var.meta,
            long_label="Wet geometric standard deviation",
            short_label="$\\sigma_{g,wet}$",
        )
    else:
        var.meta = replace(
            var.meta,
            long_label="Dry geometric standard deviation",
            short_label="$\\sigma_{g,dry}$",
        )

    return var