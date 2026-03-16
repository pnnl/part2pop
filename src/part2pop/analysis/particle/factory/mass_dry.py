from __future__ import annotations

import numpy as np

from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable


@register_particle_variable("mass_dry")
class MassDry(ParticleVariable):
    meta = VariableMeta(
        name="mass_dry",
        description="dry particle mass excluding water",
        units="kg",
        axis_names=("particle",),
        default_cfg={},
        aliases=("dry_mass",),
        scale="log",
        short_label="m_{dry}",
        long_label="dry particle mass",
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)

    def compute_from_particle(self, particle):
        return float(particle.get_mass_dry())

    def compute_all(self, population):
        return np.array(
            [population.get_particle(part_id).get_mass_dry() for part_id in population.ids],
            dtype=float,
        )


def build(cfg=None):
    cfg = cfg or {}
    return MassDry(cfg)
