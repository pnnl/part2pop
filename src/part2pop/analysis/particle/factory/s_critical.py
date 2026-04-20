from __future__ import annotations

import numpy as np

from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable


@register_particle_variable("s_critical")
class SCritical(ParticleVariable):
    meta = VariableMeta(
        name="s_critical",
        description="critical supersaturation",
        units="%",
        axis_names=("particle",),
        default_cfg={"T": 293.15},
        aliases=("s_c", "critical_supersaturation"),
        scale="log",
        short_label="s_c",
        long_label="critical supersaturation",
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)

    def compute_from_particle(self, particle):
        T = self.cfg.get("T", 293.15)
        return float(particle.get_critical_supersaturation(T=T, return_D_crit=False))

    def compute_all(self, population):
        T = self.cfg.get("T", 293.15)
        return np.array(
            [
                population.get_particle(part_id).get_critical_supersaturation(
                    T=T,
                    return_D_crit=False,
                )
                for part_id in population.ids
            ],
            dtype=float,
        )


def build(cfg=None):
    cfg = cfg or {}
    return SCritical(cfg)
