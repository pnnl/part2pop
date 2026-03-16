from __future__ import annotations

import numpy as np

from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable


@register_particle_variable("mass_tot")
class MassTot(ParticleVariable):
    meta = VariableMeta(
        name="mass_tot",
        description="total particle mass including water",
        units="kg",
        axis_names=("particle",),
        default_cfg={},
        aliases=("total_mass",),
        scale="log",
        short_label="m_{tot}",
        long_label="total particle mass",
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)

    def compute_from_particle(self, particle):
        return float(particle.get_mass_tot())

    def compute_all(self, population):
        return np.array(
            [population.get_particle(part_id).get_mass_tot() for part_id in population.ids],
            dtype=float,
        )


def build(cfg=None):
    cfg = cfg or {}
    return MassTot(cfg)
