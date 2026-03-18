from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from part2pop.freezing.builder import build_freezing_population
from part2pop.analysis.distributions import (
    make_edges,
    density1d_from_samples,
    density1d_cdf_map,
    kde1d_in_measure,
)

@register_variable("INSA_distribution")
class InsaDistVar(PopulationVariable):
    meta = VariableMeta(
        name="INSA_distribution",
        axis_names=("INSA","dN/dlnINSA"),
        description="Ice nucleating surface area distribution dN/dlnINSA.",
        units="m$^{-1}$",
        short_label="$INSA$",
        long_label="dN/dlnINSA",
        scale="linear",
        # axis/grid defaults are centralized in analysis.defaults; keep other defaults
        default_cfg={},
    )

    def compute(self, population, as_dict: bool = False):
        """
        Compute the number ice nucleating surface area distribution dN/dlnINSA on a surface area grid.

        Parameters
        ----------
        population : FreezingPopulation
            Freezing population object with particle ice nucleating surface area and number concentrations.
        as_dict : bool, optional
            If False (default), return the dN/dlnINSA array only.
            If True, return a dict with:
                - "INSA": surface area grid [m^2]
                - "dNdlnINSA": number surface area distribution [#/m^3 per ln INSA]
                - "edges": surface area bin edges [m^2]

        Notes
        -----
        - The 'method' config controls how the density is obtained:
            * "hist": conservative histogram in ln(D) using density1d_from_samples
            * "kde" : KDE in ln-space using kde1d_in_measure

        - The variable is *always* defined w.r.t. ln(INSA), i.e. measure="ln".
        """
        cfg = self.cfg
        method = cfg.get("method", "kde")
        measure = "ln"  # this variable is per dlnD by definition
        morphology = cfg.get("morphology", "homogeneous")
        
        # temperature does not matter for this plot, default to 253 K
        freezing_config={"morphology": morphology,
                         "T_grid": [253],
                         "T_units": "K"}
        freezing_pop = build_freezing_population(population, freezing_config)

        # ------------------------------------------------------------------
        # 1. Gather particle INSA and weights
        # ------------------------------------------------------------------
        INSAs = np.asarray(freezing_pop.INSA[0], dtype=float)
        weights = np.asarray(freezing_pop.num_concs, dtype=float)

        # ------------------------------------------------------------------
        # 2. Build or infer the surface area grid and edges
        # ------------------------------------------------------------------
        edges = cfg.get("edges")
        INSA_grid = cfg.get("insa_grid")
        normalize=cfg.get("normalize", False)
        
        if edges is None:
            if INSA_grid is None:
                # Default to log-spaced diameters over [INSA_min, INSA_max]
                INSA_min = cfg.get("INSA_min", 1e-15)
                INSA_max = cfg.get("INSA_max", 1e-3)
                N_bins = cfg.get("N_bins", 50)
                edges, INSA_grid = make_edges(INSA_min, INSA_max, N_bins, scale="log")
            else:
                # Infer geometric edges from center grid
                INSA_grid = np.asarray(INSA_grid, dtype=float)
                if INSA_grid.ndim != 1 or INSA_grid.size < 2:
                    raise ValueError("cfg['INSA_grid'] must be 1D with at least 2 elements.")
                r = np.sqrt(INSA_grid[1:] / INSA_grid[:-1])
                edges = np.empty(INSA_grid.size + 1, dtype=float)
                edges[1:-1] = INSA_grid[:-1] * r
                edges[0] = INSA_grid[0] / r[0]
                edges[-1] = INSA_grid[-1] * r[-1]
        else:
            edges = np.asarray(edges, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("cfg['edges'] must be 1D with at least 2 elements.")
            if np.any(edges <= 0.0):
                raise ValueError("INSA edges must be positive for dN/dlnINSA.")

            # If no explicit diameter centers were given, use geometric centers
            if INSA_grid is None:
                INSA_grid = np.sqrt(edges[:-1] * edges[1:])
            else:
                INSA_grid = np.asarray(INSA_grid, dtype=float)

        # ------------------------------------------------------------------
        # 3. Compute the density according to the chosen method
        # ------------------------------------------------------------------
        if method == "hist":
            # histogram in ln-space using the helper.  INSA_grid must already be defined.
            centers, dens, _edges = density1d_from_samples(
                INSAs,
                weights,
                edges,
                measure=measure,  # "ln"
                normalize=normalize,
            )
            INSA_grid = centers
        
        elif method == "kde":
            # KDE in ln-space using the helper.  INSA_grid must already be defined.
            INSA_grid = np.asarray(INSA_grid, dtype=float)
            dens = kde1d_in_measure(
                INSAs,
                weights,
                INSA_grid,
                measure=measure,
                normalize=normalize,
            )
        else:
            raise ValueError(f"Unknown dNdlnINSA method '{method}'")
        
        if normalize:
            dens/=np.max(dens)
        
        # ------------------------------------------------------------------
        # 4. Return in the standard analysis-variable structure
        # ------------------------------------------------------------------
        out = {
            "INSA": np.asarray(INSA_grid, dtype=float),
            "dNdlnINSA": np.asarray(dens, dtype=float),
            "edges": np.asarray(edges, dtype=float),
        }
        
        return out if as_dict else out["dNdlnINSA"]

def build(cfg=None) -> InsaDistVar:
    var = InsaDistVar(cfg or {})
    norm = var.cfg.get("normalize", False)
    if norm:
        # Normalized so that max value = 1, so unitless.
        var.meta.units = ""
        var.meta.long_label = "Normalized dN/dlnINSA"
        var.meta.short_label = r"Normalized $dN/d\ln INSA$"
    return var
