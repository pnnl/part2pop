from __future__ import annotations
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_population
import numpy as np

@register_variable("frozen_frac")
class FrozenFraction(PopulationVariable):
    meta = VariableMeta(
        name="frozen_frac",
        description="Fraction of frozen particles as a function of time for a given temperature.",
        units = r'',
        axis_names=("time_grid"),
        default_cfg={},
        aliases = ("F_frz",),
        scale = "linear",
        short_label = "frozen_frac",
        long_label = "frozen fraction",
    )
    
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        species_modifications = cfg.get("species_modifications", None)
        morphology = cfg.get("morphology", "homogeneous")
        T = cfg.get("T", None)
        RH = cfg.get("RH", None)
        T_units = cfg.get("T_units", "K")
        if not T:
            raise ValueError("Need to specify T in cfg['var_cfg'] when plotting unfrozen fraction.")
        if T_units not in ("C","K"):
            raise ValueError(f"Unknown temperature unit: '{T_units}'.")
        if T <= 0 and T_units == "K":
             raise ValueError(f"T provided is <= 0 K.")
        if T_units == "C":
             T += 273.15

        # override the underlying population species_modifications if one is supplied
        if species_modifications:
            population.species_modifications = species_modifications
        else:
            species_modifications = population.species_modifications

        # equilibrate population to RH if provided
        if RH:
            if T_units=="C":
                 population._equilibrate_h2o(RH, T+273.15)
            else:
                population._equilibrate_h2o(RH, T)

        # make freezing population
        freezing_config={"morphology": morphology,
                         #"T_grid": np.array([T]),
                         #"T_units": T_units,
                         "species_modifications": species_modifications}
        freezing_pop = build_freezing_population(population, freezing_config)
        t_min = cfg.get("t_min", 0.0)
        t_max = cfg.get("t_max", 360.0)
        dt = cfg.get("dt", 0.5)
        stochastic = cfg.get("stochastic", False)
        time = np.arange(t_min, t_max+dt, dt)
        FrozenFrac_PerPart = np.zeros((len(time), freezing_pop.num_concs.shape[0]))
        P_frz = freezing_pop.get_freezing_probs(T, freezing_config, dt=dt)    
        if stochastic:
            for t in range (1, len(time)):
                    k = np.random.binomial(100, P_frz)  # shape like P_frz
                    FrozenFrac_PerPart[t] = FrozenFrac_PerPart[t-1] + (1.0 - FrozenFrac_PerPart[t-1]) * (k / float(100))
        else:
            for t in range (1, len(time)):
                    FrozenFrac_PerPart[t] = FrozenFrac_PerPart[t-1] + P_frz * (1.0 - FrozenFrac_PerPart[t-1])
        

        FrozenFrac_population = np.sum(population.num_concs*FrozenFrac_PerPart, axis=1)/np.sum(population.num_concs)

        if as_dict:
            return {"time": np.asarray(time), "frozen_fraction": np.asarray(FrozenFrac_population)}
        return FrozenFrac_population
    
    
def build(cfg=None):
    cfg = cfg or {}
    return FrozenFraction(cfg)
