from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable

# todo: move this to a separate "distribution" variable module
# - allow different binning methods (hist, kde, etc)
# - allow different variables (Dwet, tkappa, Cabs, etc.), defined arbitrarily with ParticleVariable
# - allow different weights (dN/dlnD, dmass_bc/dlnD, dCabs/dlnD, etc.)
# - allow 1d, 2d, etc. distributions

@register_variable("dNdlnD_old")
class DNdlnDVar_old(PopulationVariable):
    meta = VariableMeta(
        name="dNdlnD",
        axis_names=("D",),
        description="Size distribution dN/dlnD",
        units="m$^{-3}$",
        scale='linear', # dN/dlnD is typically shown on linear scale; diameter itself on log scale
        long_label='Number size distribution',
        short_label='$dN/d\ln D$',
        # diameter grid default centralized in analysis.defaults; keep distribution options
        # default_cfg={
        #     "wetsize": True,
        #     "normalize": False,
        #     "method": "hist",
        #     "N_bins": 80,
        #     "D_min": 1e-9,
        #     "D_max": 2e-6,
        #     "diam_scale": "log",
        # },
    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        method = cfg.get("method", "hist")
        if method not in ("hist","kde","direct"):
            raise ValueError(f"dNdlnD method '{method}' not recognized")
        
        particles = [population.get_particle(pid) for pid in population.ids]
        if cfg["wetsize"]:
            Ds = [p.get_Dwet() for p in particles]
        else:
            Ds = [p.get_Ddry() for p in particles]
        Ds = np.asarray(Ds)

        weights = np.asarray(population.num_concs, dtype=float)
        if cfg.get("normalize",False) and weights.sum() > 0:
            weights = weights / weights.sum()
        
        if method in ("hist","kde"):
            # fixme: move these pieces into a 1D distribution utility function
            diam_grid = cfg.get("diam_grid")
            if diam_grid == None:
                D_min = cfg.get("D_min", 1e-9)
                D_max = cfg.get("D_max", 2e-6)
                N_bins = cfg.get("N_bins", 50)
                edges = np.logspace(np.log10(D_min), np.log10(D_max), N_bins + 1)
                diam_grid = np.sqrt(edges[:-1] * edges[1:])
        else:
            diam_grid = Ds

        if method == "hist":
            hist, _ = np.histogram(Ds, bins=edges, weights=weights)
            dlnD = np.log(edges[1:]) - np.log(edges[:-1])
            with np.errstate(divide="ignore", invalid="ignore"):
                dNdlnD = np.where(dlnD > 0, hist / dlnD, 0.0)
        elif method == "kde":
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(np.log(Ds), weights=weights)
            dNdlnD = kde(np.log(diam_grid))  # convert from dN/dD to dN/dlnD
        elif method == "direct":
            # direct summation of delta functions (mostly for testing)
            
            dNdlnD = np.zeros_like(diam_grid)
            for d, w in zip(Ds, weights):
                idx = np.argmin(np.abs(diam_grid - d))
                dNdlnD[idx] += w
        
        if as_dict:
            return {"D": diam_grid, "dNdlnD": dNdlnD}
        else:
            return dNdlnD


def build(cfg=None):
    """Module-level builder for population-style discovery.

    Returns an instantiated variable ready to compute.
    """
    cfg = cfg or {}
    if cfg.get("normalize") is None:
        normalize = False
    else:
        normalize = bool(cfg.get("normalize"))
    if cfg.get("wetsize") is None:
        wetsize = True
    else:
        wetsize = bool(cfg.get("wetsize"))
    
    var = DNdlnDVar(cfg)
    if normalize:
        var.meta.units = ""
    
    if not wetsize:
        var = DNdlnDVar(cfg)
        var.meta.long_label = "Dry number size distribution"
        var.meta.short_label = "$dN/d\ln D_{dry}$"
    
    return var
    #return DNdlnDVar(cfg)
