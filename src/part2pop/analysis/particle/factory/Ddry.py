from __future__ import annotations

import numpy as np

from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable


@register_particle_variable("Ddry")
class Ddry(ParticleVariable):
    meta = VariableMeta(
        name="Ddry",
        description="particle dry diameter",
        units="m",
        axis_names=("particle",),
        default_cfg={},
        aliases=("dry_diameter",),
        scale="log",
        short_label="D_{dry}",
        long_label="dry diameter",
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)

    def compute_from_particle(self, particle):
        return float(particle.get_Ddry())

    def compute_all(self, population):
        return np.array(
            [population.get_particle(part_id).get_Ddry() for part_id in population.ids],
            dtype=float,
        )


def build(cfg=None):
    cfg = cfg or {}
    return Ddry(cfg)
