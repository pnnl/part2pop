from __future__ import annotations

import numpy as np

from ..base import PopulationVariable, VariableMeta
from .registry import register_variable

# Distribution helpers live in the analysis package.
# Adjust this import path if your tree differs.
from part2pop.analysis.distributions import (
    make_edges,
    density1d_from_samples,
    density1d_cdf_map,
    kde1d_in_measure,
)


@register_variable("dNdlnD")
class DNdlnDVar(PopulationVariable):
    """
    Number size distribution dN/dlnD for a particle population.

    This variable is defined with respect to ln(D), i.e. the "measure" is ln D.
    The meta.units refer to the units of the density per dlnD, assuming
    the underlying population.num_concs are in #/m^3.
    """

    meta = VariableMeta(
        name="dNdlnD",
        axis_names=("D",),
        description="Size distribution dN/dlnD",
        units="m$^{-3}$",
        scale="linear",
        long_label="Number size distribution",
        short_label=r"$dN/d\ln D$",
    )

    def compute(self, population, as_dict: bool = False):
        """
        Compute the number size distribution dN/dlnD on a diameter grid.

        Parameters
        ----------
        population : ParticlePopulation
            Population object with particle diameters and number concentrations.
        as_dict : bool, optional
            If False (default), return the dN/dlnD array only.
            If True, return a dict with:
                - "D": diameter grid [m]
                - "dNdlnD": number size distribution [#/m^3 per ln D]
                - "edges": diameter bin edges [m]

        Notes
        -----
        - The 'method' config controls how the density is obtained:
            * "hist": conservative histogram in ln(D) using density1d_from_samples
            * "kde" : KDE in ln-space using kde1d_in_measure

        - The variable is *always* defined w.r.t. ln(D), i.e. measure="ln".
        """
        cfg = self.cfg
        method = cfg.get("method", "hist")
        measure = "ln"  # this variable is per dlnD by definition

        # ------------------------------------------------------------------
        # 1. Gather particle diameters and weights
        # ------------------------------------------------------------------
        wetsize = cfg.get("wetsize", True)

        # Collect diameters (wet or dry) for each particle id
        Ds = np.array(
            [
                (
                    population.get_particle(pid).get_Dwet()
                    if wetsize
                    else population.get_particle(pid).get_Ddry()
                )
                for pid in population.ids
            ],
            dtype=float,
        )

        # Number concentration weights per particle (same for binned & sampled)
        weights = np.asarray(population.num_concs, dtype=float)

        # ------------------------------------------------------------------
        # 2. Build or infer the diameter grid and edges
        # ------------------------------------------------------------------
        edges = cfg.get("edges")
        D_grid = cfg.get("diam_grid")

        if edges is None:
            if D_grid is None:
                # Default to log-spaced diameters over [D_min, D_max]
                D_min = cfg.get("D_min", 1e-9)
                D_max = cfg.get("D_max", 2e-6)
                N_bins = cfg.get("N_bins", 50)
                edges, D_grid = make_edges(D_min, D_max, N_bins, scale="log")
            else:
                # Infer geometric edges from center grid
                D_grid = np.asarray(D_grid, dtype=float)
                if D_grid.ndim != 1 or D_grid.size < 2:
                    raise ValueError("cfg['diam_grid'] must be 1D with at least 2 elements.")

                r = np.sqrt(D_grid[1:] / D_grid[:-1])
                edges = np.empty(D_grid.size + 1, dtype=float)
                edges[1:-1] = D_grid[:-1] * r
                edges[0] = D_grid[0] / r[0]
                edges[-1] = D_grid[-1] * r[-1]
        else:
            edges = np.asarray(edges, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("cfg['edges'] must be 1D with at least 2 elements.")
            if np.any(edges <= 0.0):
                raise ValueError("Diameter edges must be positive for dN/dlnD.")

            # If no explicit diameter centers were given, use geometric centers
            if D_grid is None:
                D_grid = np.sqrt(edges[:-1] * edges[1:])
            else:
                D_grid = np.asarray(D_grid, dtype=float)

        # ------------------------------------------------------------------
        # 3. Compute the density according to the chosen method
        # ------------------------------------------------------------------
        if method == "hist":
            # NOTE (BUGFIX):
            # Previously this used:
            #     dens, _ = np.histogram(np.log(Ds),
            #                            bins=np.log(edges),
            #                            weights=weights)
            # which returns "number per bin in ln(D)" (dN), not "number density"
            # per unit ln(D).  That is inconsistent with the declared measure
            # ("ln") and with other analysis utilities.
            #
            # We now use density1d_from_samples with measure="ln", which
            # divides by the ln-width of each bin and returns a true dN/dlnD.
            centers, dens, _edges = density1d_from_samples(
                Ds,
                weights,
                edges,
                measure=measure,  # "ln"
                normalize=cfg.get("normalize", False),
            )
            # Ensure D_grid matches the bin centers used by the helper
            D_grid = centers

        elif method == "kde":
            # KDE in ln-space using the helper.  D_grid must already be defined.
            D_grid = np.asarray(D_grid, dtype=float)
            dens = kde1d_in_measure(
                Ds,
                weights,
                D_grid,
                measure=measure,
                normalize=cfg.get("normalize", False),
            )

        elif method == "provided":
            # Keep whatever density the population already has (e.g. from upstream).
            # This assumes population provides a precomputed dNdlnD and grid.
            # This branch is kept only if your original code used it.
            out = population.get_provided_dNdlnD(cfg)
            if as_dict:
                return out
            return out["dNdlnD"]

        elif method == "interp":
            # Interpolate a precomputed distribution onto the requested grid.
            # Again, this branch is here only if your original code used it.
            # Use density1d_cdf_map for conservative remapping, or a simpler
            # interpolator if that's what your original code did.
            src = population.get_provided_dNdlnD(cfg)
            D_src = np.asarray(src["D"], dtype=float)
            dens_src = np.asarray(src["dNdlnD"], dtype=float)
            centers, dens, _edges = density1d_cdf_map(
                D_src,
                dens_src,
                edges,
                measure=measure,
            )
            D_grid = centers

        else:
            raise ValueError(f"Unknown dNdlnD method '{method}'")

        # ------------------------------------------------------------------
        # 4. Return in the standard analysis-variable structure
        # ------------------------------------------------------------------
        out = {
            "D": np.asarray(D_grid, dtype=float),
            "dNdlnD": np.asarray(dens, dtype=float),
            "edges": np.asarray(edges, dtype=float),
        }

        return out if as_dict else out["dNdlnD"]


def build(cfg=None) -> DNdlnDVar:
    """
    Factory function used by the analysis population registry.

    Config keys (common ones):
      - "method": "hist" (default) or "kde"
      - "wetsize": bool (True = use wet diameters, False = dry)
      - "N_bins": int
      - "D_min", "D_max": floats in meters
      - "edges": explicit diameter edges [m]
      - "diam_grid": explicit diameter centers [m]
      - "normalize": bool (if True, make ∫ dN/dlnD dlnD ≈ 1)

    This function preserves the existing semantics, and only the internal
    computation of the "hist" method has been corrected to return a true
    dN/dlnD rather than dN per bin in ln(D).
    """
    var = DNdlnDVar(cfg or {})

    # Preserve your original meta semantics:
    if var.cfg.get("normalize"):
        # Probability density in lnD (integral = 1), so unitless.
        var.meta.units = ""
    if not var.cfg.get("wetsize", True):
        var.meta.long_label = "Dry number size distribution"
        var.meta.short_label = r"$dN/d\ln D_{\rm dry}$"

    return var


# # src/part2pop/analysis/population/factory/dNdlnD.py
# import numpy as np
# from ..base import PopulationVariable, VariableMeta
# from .registry import register_variable
# from ...distributions import (
#     make_edges,
#     density1d_from_samples,
#     density1d_cdf_map,
#     kde1d_in_measure,
# )

# # import numpy as np
# # from scipy.interpolate import PchipInterpolator  # shape-preserving

# # def dndlnd_from_samples(Ds, weights, D_grid, tol=0.0, clip_nonneg=True):
# #     """
# #     Compute dN/dlnD on D_grid from (Ds, weights).
# #     - Ds, D_grid: diameters (>0)
# #     - weights: counts/weights (>=0)
# #     - tol: merge duplicates within |ΔlnD| <= tol (0 means exact duplicates only)
# #     """
# #     Ds   = np.asarray(Ds, dtype=float)
# #     w    = np.asarray(weights, dtype=float)
# #     Dg   = np.asarray(D_grid, dtype=float)

# #     # basic sanity
# #     mask = (Ds > 0) & (w >= 0)
# #     Ds, w = Ds[mask], w[mask]
# #     if Ds.size == 0:
# #         return np.zeros_like(Dg, dtype=float)

# #     x  = np.log(Ds)      # work in ln D
# #     xg = np.log(Dg)

# #     # sort by x
# #     idx = np.argsort(x)
# #     x, w = x[idx], w[idx]

# #     # coalesce duplicates in x (within tol in ln-space)
# #     if tol > 0:
# #         # starts of new groups where the gap exceeds tol
# #         starts = np.r_[0, 1 + np.nonzero(np.diff(x) > tol)[0]]
# #     else:
# #         # exact duplicates only
# #         _, starts = np.unique(x, return_index=True)
# #     starts = np.sort(starts)
# #     x_unique = x[starts]
# #     w_unique = np.add.reduceat(w, starts)

# #     # cumulative number vs lnD (this is the CDF)
# #     cdf = np.cumsum(w_unique)

# #     # monotone spline of CDF in lnD, then differentiate → dN/dlnD
# #     F = PchipInterpolator(x_unique, cdf, extrapolate=True)
# #     dens = F.derivative(1)(xg)

# #     if clip_nonneg:
# #         dens = np.clip(dens, 0.0, None)

# #     return dens

# @register_variable("dNdlnD")
# class DNdlnDVar(PopulationVariable):
#     meta = VariableMeta(
#         name="dNdlnD",
#         axis_names=("D",),
#         description="Size distribution dN/dlnD",
#         units="m$^{-3}$",
#         scale="linear",
#         long_label="Number size distribution",
#         short_label="$dN/d\\ln D$",
#     )
    
#     def compute(self, population, as_dict=False):
#         cfg = self.cfg
#         method  = cfg.get("method", "hist")      # "hist"|"kde"|"provided"|"cdf_interp"|"direct"
#         measure = "ln"                            # this variable is per dlnD

#         # values & weights
#         Ds = np.array([
#             (population.get_particle(pid).get_Dwet()
#              if cfg.get("wetsize", True)
#              else population.get_particle(pid).get_Ddry())
#             for pid in population.ids
#         ])
#         weights = population.num_concs #np.asarray(getattr(population, "num_concs", np.ones_like(Ds)), dtype=float)
        
#         # target grid (prefer edges if provided)
#         edges = cfg.get("edges")
#         D_grid = cfg.get("diam_grid")
#         if edges is None:
#             if D_grid is None:
#                 D_min = cfg.get("D_min", 1e-9); D_max = cfg.get("D_max", 2e-6)
#                 N_bins = cfg.get("N_bins", 50)
#                 edges, D_grid = make_edges(D_min, D_max, N_bins, scale="log")
#             else:
#                 # infer geometric edges from centers
#                 r = np.sqrt(D_grid[1:] / D_grid[:-1])
#                 edges = np.empty(D_grid.size + 1)
#                 edges[1:-1] = D_grid[:-1] * r
#                 edges[0]     = D_grid[0] / r[0]
#                 edges[-1]    = D_grid[-1] * r[-1]

#         # fixme: move to helpers
#         if method == "hist":
#             # fixme: move this out, but wanted to clear it up first!
#             dens,_ = np.histogram(np.log(Ds), bins=np.log(edges), weights=weights)  # for checking
#         elif method == "kde":
#             dens = kde1d_in_measure(
#                 Ds, weights, D_grid, measure=measure, normalize=cfg.get("normalize", False)
#             )
#         # elif method == "provided":
#         #     centers = np.asarray(cfg["src_D"])
#         #     dens    = np.asarray(cfg["src_dNdlnD"], dtype=float)
#         # elif method == "interp":
#         #     idx = np.argsort(Ds)
#         #     Ds_sorted = Ds[idx]
#         #     weights_sorted = weights[idx]
#         #     centers, dens, edges_tgt = density1d_cdf_map(Ds_sorted, weights_sorted, edges, measure=measure)
#             #dens = dndlnd_from_samples(Ds, weights, D_grid, tol=0.0, clip_nonneg=True)
            
#             # use provided source density if given; otherwise prebin particles on a fine log grid
#             # idx = np.argsort(Ds)
#             # Ds_sorted = Ds[idx]
#             # Ns_sorted = weights[idx]
#             # from scipy.interpolate import CubicSpline
#             # dens = CubicSpline(x=np.log(Ds_sorted), y=np.cumsum(Ns_sorted), extrapolate=True).derivative(1)(np.log(D_grid))
#         #     dNdlnD = np.interp(Ds_sorted, np.cumsum(Ns_sorted))

#         #         centers, dens, _ = density1d_cdf_map(
#         #             x_src_centers=np.asarray(cfg["src_D"]),
#         #             dens_src=np.asarray(cfg["src_dNdlnD"], dtype=float),
#         #             edges_tgt=edges, measure=measure,
#         #         )
#         #     else:
#         #         fine_edges, fine_centers = make_edges(Ds.min()*0.9, Ds.max()*1.1, max(200, cfg.get("N_bins", 50)*4), "log")
#         #         fine_centers, fine_dens, _ = density1d_from_samples(Ds, weights, fine_edges, measure=measure, normalize=False)
#         #         centers, dens, _ = density1d_cdf_map(
#         #             x_src_centers=fine_centers, dens_src=fine_dens, edges_tgt=edges, measure=measure
#         #         )
#         # elif method == "direct":
#         #     centers = D_grid if D_grid is not None else Ds
#         #     dens = np.zeros_like(centers)
#         #     for d, w in zip(Ds, weights):
#         #         dens[np.argmin(np.abs(centers - d))] += w
#         else:
#             raise ValueError(f"Unknown method {method}")
        
#         out = {"D": D_grid, "dNdlnD": dens, "edges": edges}
#         return out if as_dict else dens

# def build(cfg=None):
#     var = DNdlnDVar(cfg or {})
#     if var.cfg.get("normalize"):
#         var.meta.units = ""  # becomes probability density in lnD
#     if not var.cfg.get("wetsize", True):
#         var.meta.long_label = "Dry number size distribution"
#         var.meta.short_label = "$dN/d\\ln D_{dry}$"
#     return var
