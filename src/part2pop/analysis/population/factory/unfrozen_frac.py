from __future__ import annotations
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_population
from part2pop.analysis._freezing import resolve_freezing_config
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
        species_modifications = resolve_freezing_config(population, cfg)["species_modifications"]
        morphology = cfg.get("morphology", "homogeneous")
        T = cfg.get("T", None)
        RH = cfg.get("RH", None)
        T_units = cfg.get("T_units", "K")
        if T is None:
            raise ValueError("Need to specify T in cfg['var_cfg'] when plotting freezing fraction.")
        if T_units not in ("C", "K"):
            raise ValueError(f"Unknown temperature unit: '{T_units}'.")
        T = float(T) + (273.15 if T_units == "C" else 0.0)
        if T <= 0 or not np.isfinite(T):
            raise ValueError("T must be finite and above 0 K.")
        if RH is not None:
            RH = float(RH)
            # Validate before equilibration mutates water masses. Values above
            # 0.99 used to be clipped silently, so a percent input of 85 ran
            # at RH = 0.99.
            if not np.isfinite(RH) or RH < 0 or RH > 1:
                raise ValueError(
                    f"RH must be a fraction between 0 and 1, got {RH!r}; "
                    "give 85% as 0.85."
                )

        # Species overrides apply to this calculation, not population defaults.

        # T has already been converted to Kelvin (if necessary).
        if RH is not None:
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
            return {"time": np.asarray(time), "unfrozen_fraction": 1.0-np.asarray(FrozenFrac_population)}
        return 1.0-FrozenFrac_population
    
    
def build(cfg=None):
    cfg = cfg or {}
    return UnfrozenFraction(cfg)
