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
        default_cfg={"RH": 0.85, "T": 243.0, "T_units": "K"},
        aliases = ('P_frz',),
        scale = 'log',
        short_label = 'P_{frz}',
        long_label = 'freezing probability',
    )
    
    def compute_all(self, population):
        config = self.cfg        
        T = config.get("T", None)
        RH = config.get("RH", None)
        species_modifications = config.get("species_modifications", None)
        morphology = config.get("morphology", "homogeneous")
        T_units = config.get("T_units", "K")
        if not T:
            raise ValueError("Need to specify temperature in cfg['var_cfg'] when plotting freezing probability.")
        elif T_units not in ("C","K"):
            raise ValueError(f"Unknown temperature unit: '{T_units}'.")
        if RH:
            if T_units=="C":
                 population._equilibrate_h2o(RH, T+273.15)
            else:
                population._equilibrate_h2o(RH, T)

        # override the underlying population species_modifications if one is supplied
        if species_modifications:
            population.species_modifications = species_modifications
        else:
            species_modifications = population.species_modifications

        # make freezing population
        freezing_config={"morphology": morphology,
                         "T_grid": np.array([T]),
                         "T_units": T_units,
                         "species_modifications": species_modifications}
        freezing_pop = build_freezing_population(population, freezing_config)
        freezing_probs = freezing_pop.get_freezing_probs()
        return freezing_probs
    
    
def build(cfg=None):
    cfg = cfg or {}
    return FreezingProb(cfg)
