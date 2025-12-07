from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.optics.builder import build_optical_population

# fixme: seems like a janky way to handle this
@register_variable("wvl_grid")
class WvlGridVar(PopulationVariable):
    meta = VariableMeta(
        name="wvl_grid",
        axis_names=(),
        description="Wavelength",
        units="m",
        short_label = '$\lambda$',
        long_label = 'wavelength',
        scale='linear',
        # axis/grid defaults are centralized in analysis.defaults
        default_cfg={},
        aliases=('wvls',),
    )
    def compute(self, population=None,as_dict=False):
        cfg = self.cfg
        if isinstance(cfg.get("wvl_grid"), (list, tuple, np.ndarray)):
            arr = np.asarray(cfg["wvl_grid"])
        else:
            arr = np.asarray([])
        if as_dict:
            return {"wvl_grid": arr}
        return arr
    # fixme: add scale units

def build(cfg=None):
    cfg = cfg or {}
    return WvlGridVar(cfg)
