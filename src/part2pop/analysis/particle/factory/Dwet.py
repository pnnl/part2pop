from __future__ import annotations
from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable
import numpy as np

@register_particle_variable("Dwet")

class Dwet(ParticleVariable):
    meta = VariableMeta(
        name="Dwet",
        description='particle wet diameter',
        units = 'm',
        axis_names=("rh_grid"),
        default_cfg={},
        aliases = ('wet_diameter',),
        scale = 'log',
        short_label = 'D_{wet}',
        long_label = 'wet diameter',
    )

    def compute_one(self, population, part_id):
        particle = population.get_particle(part_id)
        return self.compute_from_particle(particle)
    
    def compute_from_particle(self, particle):
        return particle.get_Dwet()
    
    def compute_all(self, population):
        return np.array([population.get_particle(part_id).get_Dwet() for part_id in population.ids])
    
def build(cfg=None):
    cfg = cfg or {}
    return Dwet(cfg)
