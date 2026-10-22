from __future__ import annotations
from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable
from part2pop.freezing.builder import build_freezing_population
from part2pop.analysis._freezing import resolve_freezing_config
import numpy as np
from part2pop.analysis._humidity import validate_equilibration_rh

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
        RH = validate_equilibration_rh(config.get("RH", 0.85))
        species_modifications = resolve_freezing_config(population, config)["species_modifications"]
        morphology = config.get("morphology", "homogeneous")
        T_units = config.get("T_units", "K")
        if T is None:
            raise ValueError("Need to specify temperature in cfg['var_cfg'] when plotting freezing probability.")
        if T_units not in ("C", "K"):
            raise ValueError(f"Unknown temperature unit: '{T_units}'.")
        T = float(T) + (273.15 if T_units == "C" else 0.0)
        if not np.isfinite(T) or T <= 0:
            raise ValueError("Temperature must be finite and above 0 K.")
        population._equilibrate_h2o(RH, T)

        # Species overrides apply to this calculation, not population defaults.

        # make freezing population
        freezing_config={"morphology": morphology,
                         "species_modifications": species_modifications}
        freezing_pop = build_freezing_population(population, freezing_config)
        freezing_probs = freezing_pop.get_freezing_probs(T, freezing_config)        
        return freezing_probs
    
    
def build(cfg=None):
    cfg = cfg or {}
    return FreezingProb(cfg)
