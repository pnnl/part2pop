from __future__ import annotations
from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable
import numpy as np

@register_particle_variable("kappa")

class kappa(ParticleVariable):
    meta = VariableMeta(
        name="kappa",
        description='particle hygroscopicity parameter',
        units = None,
        axis_names=("rh_grid"),
        default_cfg={},
        aliases = ('kappa',),
        scale = 'log',
        short_label = 'k',
        long_label = 'hygroscopicity parameter',
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)
    
    def compute_from_particle(self, particle):
        return particle.get_tkappa()
    
    def compute_all(self, population):
        return np.array([population.get_particle(part_id).get_tkappa() for part_id in population.ids])
    
def build(cfg=None):
    cfg = cfg or {}
    return kappa(cfg)
