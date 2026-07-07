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


@register_variable("geom_mean_diameter")
class GeomMeanDiameterVar(PopulationVariable):
    meta = VariableMeta(
        name="geom_mean_diameter",
        axis_names=(),
        description="Number-weighted geometric mean particle diameter",
        units="m",
        scale="linear",
        long_label="Geometric mean diameter",
        short_label="$D_g$",
        default_cfg={"wetsize": True},
        aliases=(
            "geometric_mean_diameter",
            "geom_mean_diam",
            "GMD",
            "Dg",
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
            value = float(
                np.exp(np.sum(weights * np.log(diameters)) / np.sum(weights))
            )

        if as_dict:
            return {"geom_mean_diameter": value}

        return value


def build(cfg=None):
    cfg = cfg or {}
    var = GeomMeanDiameterVar(cfg)

    if cfg.get("wetsize", True):
        var.meta = replace(
            var.meta,
            long_label="Wet geometric mean diameter",
            short_label="$D_{g,wet}$",
        )
    else:
        var.meta = replace(
            var.meta,
            long_label="Dry geometric mean diameter",
            short_label="$D_{g,dry}$",
        )

    return var