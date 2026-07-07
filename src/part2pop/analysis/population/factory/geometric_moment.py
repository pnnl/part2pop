from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..base import PopulationVariable, VariableMeta
from .registry import register_variable


def _moment_power(cfg):
    if "moment_power" in cfg:
        power = cfg["moment_power"]
    elif "power" in cfg:
        power = cfg["power"]
    elif "k" in cfg:
        power = cfg["k"]
    else:
        power = 0.0

    power = float(power)

    if not np.isfinite(power):
        raise ValueError("Moment power must be finite.")

    return power


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


def _format_power(power: float) -> str:
    return f"{power:g}"


def _moment_units(power: float, *, normalize: bool) -> str:
    power_text = _format_power(power)

    if power == 0:
        return "" if normalize else "m$^{-3}$"

    diameter_units = "m" if power == 1 else f"m$^{{{power_text}}}$"

    if normalize:
        return diameter_units

    return f"{diameter_units} m$^{{-3}}$"


@register_variable("geometric_moment")
class GeometricMomentVar(PopulationVariable):
    meta = VariableMeta(
        name="geometric_moment",
        axis_names=(),
        description="Number-concentration-weighted particle diameter moment",
        units="m$^{-3}$",
        scale="linear",
        long_label="Diameter moment",
        short_label="$M_k$",
        default_cfg={
            "moment_power": 0.0,
            "wetsize": True,
            "normalize": False,
        },
        aliases=(
            "diameter_moment",
            "D_moment",
            "moment_D",
        ),
    )

    def compute(self, population, as_dict: bool = False):
        power = _moment_power(self.cfg)
        normalize = bool(self.cfg.get("normalize", False))

        diameters, weights = _diameters_and_weights(
            population,
            wetsize=bool(self.cfg.get("wetsize", True)),
        )

        if diameters.size == 0:
            value = np.nan
        else:
            value = float(np.sum(weights * diameters**power))

            if normalize:
                total_weight = float(np.sum(weights))
                value = value / total_weight if total_weight > 0.0 else np.nan

        if as_dict:
            return {
                "geometric_moment": value,
                "moment_power": power,
                "wetsize": bool(self.cfg.get("wetsize", True)),
                "normalize": normalize,
            }

        return value


def build(cfg=None):
    cfg = cfg or {}
    var = GeometricMomentVar(cfg)

    power = _moment_power(var.cfg)
    power_text = _format_power(power)
    normalize = bool(var.cfg.get("normalize", False))
    basis = "wet" if var.cfg.get("wetsize", True) else "dry"
    units = _moment_units(power, normalize=normalize)

    if normalize:
        long_label = f"Mean {basis} diameter moment"
        short_label = f"$\\langle D^{{{power_text}}}\\rangle$"
    else:
        long_label = f"{basis.capitalize()} diameter moment"
        short_label = f"$M_{{{power_text}}}$"

    var.meta = replace(
        var.meta,
        units=units,
        long_label=long_label,
        short_label=short_label,
    )

    return var