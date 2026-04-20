from __future__ import annotations
from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable
from part2pop.freezing.builder import build_freezing_population
import numpy as np

@register_particle_variable("P_frz")
class FreezingProb(ParticleVariable):
    meta = VariableMeta(
        name="P_frz",
        description='Probability that a particle will freeze over 1 s.',
        units = 'over 1s time step',
        axis_names=("T_grid"),
        default_cfg={},
        aliases = ('P_frz',),
        scale = 'log',
        short_label = 'P_{frz}',
        long_label = 'freezing probability',
    )
    
    def compute_all(self, population):
        cfg = self.cfg
        if not cfg["T"]:
            raise ValueError("Need to specify temperature in cfg['var_cfg'] when plotting freezing probability.")
        freezing_cfg = {
            "T_grid": list(np.array([cfg["T"]])),
            "morphology": cfg.get("morphology", "homogeneous"),
            "species_modifications": cfg.get("species_modifications", {}),
            "T_units": cfg.get("T_units", "K")
        }
        freezing_pop = build_freezing_population(population, freezing_cfg) 
        return freezing_pop.get_freezing_probs()[0]
    
    
def build(cfg=None):
    cfg = cfg or {}
    return FreezingProb(cfg)
