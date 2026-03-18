from __future__ import annotations
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
import numpy as np

@register_variable("unfrozen_frac")
class UnfrozenFraction(PopulationVariable):
    meta = VariableMeta(
        name="unfrozen_frac",
        description="Fraction of unfrozen particles for a given temperature as a function of time.",
        units = r'',
        axis_names=("time_grid"),
        default_cfg={},
        aliases = ("F_unfrz",),
        scale = "log",
        short_label = "unfrozen_frac",
        long_label = "unfrozen fraction",
    )
    
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        t_min = cfg.get("t_min", 0.0)
        t_max = cfg.get("t_max", 360.0)
        dt = cfg.get("dt", 0.5)
        stochastic = cfg.get("stochastic", False)
        time = np.arange(t_min, t_max+dt, dt)
        FrozenFrac_PerPart = np.zeros((len(time), len(population.T_grid), population.num_concs.shape[0]))
        P_frz = population.get_freezing_probs(dt=dt)        
        if stochastic:
            for t in range (1, len(time)):
                    k = np.random.binomial(100, P_frz)  # shape like P_frz
                    FrozenFrac_PerPart[t] = FrozenFrac_PerPart[t-1] + (1.0 - FrozenFrac_PerPart[t-1]) * (k / float(100))
        else:
            for t in range (1, len(time)):
                    FrozenFrac_PerPart[t] = FrozenFrac_PerPart[t-1] + P_frz * (1.0 - FrozenFrac_PerPart[t-1])
        
        FrozenFrac_population = np.sum(population.num_concs*FrozenFrac_PerPart, axis=2)/np.sum(population.num_concs)

        if as_dict:
            return {"time": np.asarray(time), "frozen_fraction": 1.0-np.asarray(FrozenFrac_population)}
        return 1.0-FrozenFrac_population
    
    
def build(cfg=None):
    cfg = cfg or {}
    return UnfrozenFraction(cfg)
