from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable

@register_variable("spec_mass_conc")
class SpecMassConc(PopulationVariable):
    meta = VariableMeta(
        name="spec_mass_conc",
        axis_names=("species"),
        description="speciated mass concentration",
        units="kg/m$^{3}$",
        aliases=(),
        scale='linear',
        long_label = 'speciated mass concentration',
        short_label = '$m$',
        # s-grid default centralized in analysis.defaults; keep other defaults
        default_cfg={"species_names": None},
    )
    
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        species_names = cfg.get("species_names", None)
        if species_names is None:
            species_names = [spec.name for spec in population.species]
        if type(species_names) is str:
            species_names = [species_names]
        
        out = np.zeros(len(species_names), dtype=float)
        for kk,spec_name in enumerate(species_names):
            out[kk] = population.get_species_mass_conc(spec_name)
        if as_dict:
            return {"species_names": species_names, "mass_conc": out}
        return out

def build(cfg=None):
    cfg = cfg or {}
    return SpecMassConc(cfg)