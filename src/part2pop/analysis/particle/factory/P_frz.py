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
        config = self.cfg
        T = config.get("T",None)
        T_units = config.get("T_units", "K")
        if not T:
            raise ValueError("Need to specify temperature in cfg['var_cfg'] when plotting freezing probability.")
        if T_units=="C":        
            T = T+273.15
        elif T_units not in ("C","K"):
            raise ValueError(f"Unknown temperature unit: '{T_units}'.")
        if T < population.T_grid.min() or T > population.T_grid.max():
            if T_units=="C":
                raise ValueError(f"T provided ({T-273.15} C) to P_frz plotter is outside of T_grid: {population.T_grid.min()-273.15} C to {population.T_grid.max()-273.15} C")
            else:
                raise ValueError(f"T provided ({T} K) to P_frz plotter is outside of T_grid: {population.T_grid.min()} K to {population.T_grid.max()} K")
        
        freezing_probs = population.get_freezing_probs()
        xp = population.T_grid
        out = np.zeros(len(population.num_concs))
        for ii in range(len(out)):
            fp = freezing_probs[:,ii]
            out[ii] = np.interp(T, xp=xp, fp=fp)
        return out
    
    
def build(cfg=None):
    cfg = cfg or {}
    return FreezingProb(cfg)
