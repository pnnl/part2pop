from __future__ import annotations

import numpy as np

from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable


@register_particle_variable("D_critical")
class DCritical(ParticleVariable):
    meta = VariableMeta(
        name="D_critical",
        description="critical diameter",
        units="m",
        axis_names=("particle",),
        default_cfg={"T": 293.15},
        aliases=("D_c", "critical_diameter"),
        scale="log",
        short_label="D_c",
        long_label="critical diameter",
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)

    def compute_from_particle(self, particle):
        T = self.cfg.get("T", 293.15)
        _, d_crit = particle.get_critical_supersaturation(T=T, return_D_crit=True)
        return float(d_crit)

    def compute_all(self, population):
        T = self.cfg.get("T", 293.15)
        out = []
        for part_id in population.ids:
            _, d_crit = population.get_particle(part_id).get_critical_supersaturation(
                T=T,
                return_D_crit=True,
            )
            out.append(d_crit)
        return np.array(out, dtype=float)


def build(cfg=None):
    cfg = cfg or {}
    return DCritical(cfg)
