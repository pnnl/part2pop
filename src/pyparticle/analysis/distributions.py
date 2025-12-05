from __future__ import annotations
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    from scipy.interpolate import RegularGridInterpolator as _RGI
except Exception:
    _PCHIP = None
    _RGI = None


# ---------- Grid helpers ----------

def make_edges(xmin: float, xmax: float, n_bins: int, scale: str = "log"):
    """Return (edges, centers) on linear or logarithmic scale."""
    if scale == "log":
        edges = np.geomspace(xmin, xmax, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    elif scale == "linear":
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        raise ValueError("scale must be 'log' or 'linear'")
    return edges, centers


def bin_widths(edges: np.ndarray, measure: str = "ln"):
    """
    Widths in the measure of integration.

    - measure == "ln":     widths = d(ln x) between edges
    - measure == "linear": widths = dx between edges
    """
    if measure == "ln":
        return np.log(edges[1:]) - np.log(edges[:-1])
    elif measure == "linear":
        return edges[1:] - edges[:-1]
    else:
        raise ValueError("measure must be 'ln' or 'linear'")


def _u_from_x(x: np.ndarray, measure: str):
    """Change of variable to the integration coordinate u."""
    if measure == "ln":
        return np.log(x)
    elif measure == "linear":
        return np.asarray(x)
    else:
        raise ValueError("measure must be 'ln' or 'linear'")


# ---------- 1D distributions ----------

def density1d_from_samples(
    x: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    measure: str = "ln",
    normalize: bool = False,
):
    """
    Conservative histogram of samples into a **number density** wrt the chosen measure.

    Parameters
    ----------
    x : array
        Sample locations (e.g., diameters).
    weights : array
        Sample weights (e.g., counts or number concentration per sample).
    edges : array
        Bin edges in x.
    measure : {"ln", "linear"}
        Measure in which the density is defined:
          - "ln":     dens is dN/d(ln x)
          - "linear": dens is dN/dx
    normalize : bool
        If True, rescale dens so that the integral
            ∫ dens d(measure) ≈ 1
        when computed via trapz over centers in u-space:
            u = ln(centers) if measure == "ln"
            u = centers     if measure == "linear"

    Returns
    -------
    centers : array
        Bin centers in x.
    dens : array
        Number density per d(measure) (e.g., dN/dlnx).
    edges : array
        The input edges, passed through.
    """
    x = np.asarray(x)
    w = np.asarray(weights, dtype=float)

    # Histogram of counts/weights per bin
    H, _ = np.histogram(x, bins=edges, weights=w)

    # Widths in the chosen measure
    widths = bin_widths(edges, measure)

    # Density per d{measure}x on each bin (cell-average): dN/d(measure)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(widths > 0, H / widths, 0.0)

    # Bin centers in x
    if measure == "linear":
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        centers = np.sqrt(edges[:-1] * edges[1:])

    if normalize:
        # IMPORTANT:
        #  - dens remains a number density (H / widths).
        #  - We normalize using the same quadrature that users/tests use:
        #      total = ∫ dens d(measure) ≈ trapz(dens, u_centers)
        #    where u_centers is ln(centers) or centers depending on measure.
        u_centers = _u_from_x(centers, measure)
        total = np.trapz(dens, u_centers)
        if total > 0:
            dens = dens / total

    return centers, dens, edges


def density1d_cdf_map(
    x_src_centers: np.ndarray,
    dens_src: np.ndarray,
    edges_tgt: np.ndarray,
    measure: str = "ln",
):
    """
    Conservative mapping of a tabulated 1D **number density** (per d{measure}x)
    onto target edges.

    Steps:
      - Build "source" edges around input centers.
      - Integrate dens_src over source cells in u-space to get a CDF.
      - Interpolate the CDF to target edges.
      - Difference to recover counts per target bin.
      - Divide by target widths to get number density on target bins.

    Returns (centers_tgt, dens_tgt, edges_tgt).
    """
    x_src = np.asarray(x_src_centers)
    y_src = np.asarray(dens_src, dtype=float)

    # Build edges around centers
    if measure == "ln":
        r = np.sqrt(x_src[1:] / x_src[:-1])
        src_edges = np.empty(x_src.size + 1)
        src_edges[1:-1] = x_src[:-1] * r
        src_edges[0] = x_src[0] / r[0]
        src_edges[-1] = x_src[-1] * r[-1]
    else:
        d = 0.5 * (x_src[1:] - x_src[:-1])
        src_edges = np.empty(x_src.size + 1)
        src_edges[1:-1] = 0.5 * (x_src[:-1] + x_src[1:])
        src_edges[0] = x_src[0] - d[0]
        src_edges[-1] = x_src[-1] + d[-1]

    # Integrate to CDF in u-space
    u_edges_src = _u_from_x(src_edges, measure)
    du_src = np.diff(u_edges_src)
    cell_N = y_src * du_src  # numbers per bin
    N_src = np.concatenate([[0.0], np.cumsum(cell_N)])  # CDF at src_edges

    u_edges_tgt = _u_from_x(edges_tgt, measure)
    if _PCHIP is not None and N_src.size >= 2:
        # monotone interpolator if available
        N_of_u = _PCHIP(u_edges_src, N_src, extrapolate=True)
        N_edges = N_of_u(u_edges_tgt)
    else:
        N_edges = np.interp(
            u_edges_tgt,
            u_edges_src,
            N_src,
            left=0.0,
            right=N_src[-1],
        )

    widths = bin_widths(edges_tgt, measure)
    dN = np.maximum(0.0, np.diff(N_edges))
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_tgt = np.where(widths > 0, dN / widths, 0.0)

    if measure == "linear":
        centers = 0.5 * (edges_tgt[:-1] + edges_tgt[1:])
    else:
        centers = np.sqrt(edges_tgt[:-1] * edges_tgt[1:])

    return centers, dens_tgt, edges_tgt


def kde1d_in_measure(
    x: np.ndarray,
    weights: np.ndarray,
    xq: np.ndarray,
    measure: str = "ln",
    normalize: bool = False,
):
    """
    Smooth estimate of **number density** wrt the chosen measure using a KDE in u-space.

    Parameters
    ----------
    x : array
        Sample locations.
    weights : array
        Sample weights.
    xq : array
        Query points in x.
    measure : {"ln", "linear"}
        - "ln":     dens is dN/d(ln x)
        - "linear": dens is dN/dx
    normalize : bool
        If True, dens is rescaled so that the integral over u (ln x or x)
        approximated by trapz(dens, u) is 1.

    Returns
    -------
    dens : array
        KDE-evaluated number density per d(measure) at xq.
    """
    try:
        from scipy.stats import gaussian_kde
    except Exception as e:
        raise RuntimeError("scipy is required for KDE") from e

    x = np.asarray(x)
    w = np.asarray(weights, dtype=float)
    xq = np.asarray(xq)

    # Work in u-space
    u = _u_from_x(x, measure)
    u_q = _u_from_x(xq, measure)

    kde = gaussian_kde(u, weights=w)
    dens = kde(u_q)  # dens(u) per du

    if normalize:
        total = np.trapz(dens, u_q)
        if total > 0:
            dens = dens / total

    return dens


# ---------- 2D distributions ----------

def density2d_from_samples(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    measure_x: str = "ln",
    measure_y: str = "ln",
    normalize: bool = False,
):
    """
    Conservative 2D histogram -> **number density** per
    d{measure_x}x d{measure_y}y.

    Returns
    -------
    centers_x, centers_y : arrays
        Bin centers in x and y.
    dens : array
        Number density per d(measure_x) d(measure_y).
    edges_x, edges_y : arrays
        The input edges, passed through.
    """
    H, ex, ey = np.histogram2d(
        x,
        y,
        bins=[edges_x, edges_y],
        weights=np.asarray(weights, dtype=float),
    )

    wx = bin_widths(ex, measure_x)[:, None]
    wy = bin_widths(ey, measure_y)[None, :]

    # dens = counts / (width_x * width_y) → dN/(dmeasure_x dmeasure_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where((wx > 0) & (wy > 0), H / (wx * wy), 0.0)

    if normalize:
        # Normalize via integrals in (u_x, u_y):
        # N_total = ∫∫ dens d(measure_x)d(measure_y) ≈ Σ dens * wx * wy
        total = (dens * wx * wy).sum()
        if total > 0:
            dens = dens / total

    if measure_x == "linear":
        cx = 0.5 * (ex[:-1] + ex[1:])
    else:
        cx = np.sqrt(ex[:-1] * ex[1:])

    if measure_y == "linear":
        cy = 0.5 * (ey[:-1] + ey[1:])
    else:
        cy = np.sqrt(ey[:-1] * ey[1:])

    return cx, cy, dens, ex, ey


def density2d_cdf_map(
    src_edges_x: np.ndarray,
    src_edges_y: np.ndarray,
    dens_src: np.ndarray,  # per d{measure_x}x d{measure_y}y on src cells
    tgt_edges_x: np.ndarray,
    tgt_edges_y: np.ndarray,
    measure_x: str = "ln",
    measure_y: str = "ln",
):
    """
    Conservative mapping of a 2D **number density** on a rectilinear source grid
    onto target edges. Uses the 2D CDF in (u_x, u_y), then inclusion-exclusion
    per target cell.
    """
    if _RGI is None:
        raise RuntimeError("scipy RegularGridInterpolator is required for 2D CDF mapping")

    ux_e_src = _u_from_x(src_edges_x, measure_x)
    uy_e_src = _u_from_x(src_edges_y, measure_y)
    dux = np.diff(ux_e_src)
    duy = np.diff(uy_e_src)

    # integrate density over source cells to get counts
    cell_N = dens_src * (dux[:, None] * duy[None, :])

    # Build CDF on edge grid (nx+1, ny+1)
    N = np.zeros((cell_N.shape[0] + 1, cell_N.shape[1] + 1))
    N[1:, 1:] = cell_N.cumsum(axis=0).cumsum(axis=1)

    # Interpolate CDF to target edge grid
    ux_e_tgt = _u_from_x(tgt_edges_x, measure_x)
    uy_e_tgt = _u_from_x(tgt_edges_y, measure_y)
    rgi = _RGI((ux_e_src, uy_e_src), N, bounds_error=False, fill_value=(N[-1, -1]))
    Ux, Uy = np.meshgrid(ux_e_tgt, uy_e_tgt, indexing="ij")
    Nt = rgi(np.stack([Ux, Uy], axis=-1))  # shape (Nx+1, Ny+1)

    # Inclusion-exclusion to recover counts per target cell
    dN = Nt[1:, 1:] - Nt[:-1, 1:] - Nt[1:, :-1] + Nt[:-1, :-1]
    dux_t = np.diff(ux_e_tgt)[:, None]
    duy_t = np.diff(uy_e_tgt)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_tgt = np.where((dux_t > 0) & (duy_t > 0), dN / (dux_t * duy_t), 0.0)

    if measure_x == "linear":
        cx = 0.5 * (tgt_edges_x[:-1] + tgt_edges_x[1:])
    else:
        cx = np.sqrt(tgt_edges_x[:-1] * tgt_edges_x[1:])

    if measure_y == "linear":
        cy = 0.5 * (tgt_edges_y[:-1] + tgt_edges_y[1:])
    else:
        cy = np.sqrt(tgt_edges_y[:-1] * tgt_edges_y[1:])

    return cx, cy, dens_tgt, tgt_edges_x, tgt_edges_y

# from __future__ import annotations
# import numpy as np

# try:
#     from scipy.interpolate import PchipInterpolator as _PCHIP
#     from scipy.interpolate import RegularGridInterpolator as _RGI
# except Exception:
#     _PCHIP = None
#     _RGI = None


# # ---------- Grid helpers ----------

# def make_edges(xmin: float, xmax: float, n_bins: int, scale: str = "log"):
#     """Return (edges, centers) on linear or logarithmic scale."""
#     if scale == "log":
#         edges = np.geomspace(xmin, xmax, n_bins + 1)
#         centers = np.sqrt(edges[:-1] * edges[1:])
#     elif scale == "linear":
#         edges = np.linspace(xmin, xmax, n_bins + 1)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#     else:
#         raise ValueError("scale must be 'log' or 'linear'")
#     return edges, centers


# def bin_widths(edges: np.ndarray, measure: str = "ln"):
#     """
#     Widths in the measure of integration.

#     - measure == "ln":   widths = d(ln x) between edges
#     - measure == "linear": widths = dx between edges
#     """
#     if measure == "ln":
#         return np.log(edges[1:]) - np.log(edges[:-1])
#     elif measure == "linear":
#         return edges[1:] - edges[:-1]
#     else:
#         raise ValueError("measure must be 'ln' or 'linear'")


# def _u_from_x(x: np.ndarray, measure: str):
#     """Change of variable to the integration coordinate u."""
#     if measure == "ln":
#         return np.log(x)
#     elif measure == "linear":
#         return np.asarray(x)
#     else:
#         raise ValueError("measure must be 'ln' or 'linear'")


# # ---------- 1D distributions ----------

# def density1d_from_samples(
#     x: np.ndarray,
#     weights: np.ndarray,
#     edges: np.ndarray,
#     measure: str = "ln",
#     normalize: bool = False,
# ):
#     """
#     Conservative histogram of samples into a **number density** wrt the chosen measure.

#     Parameters
#     ----------
#     x : array
#         Sample locations (e.g., diameters).
#     weights : array
#         Sample weights (e.g., counts or number concentration per sample).
#     edges : array
#         Bin edges in x.
#     measure : {"ln", "linear"}
#         Measure in which the density is defined:
#           - "ln":    dens is dN/d(ln x)
#           - "linear": dens is dN/dx
#     normalize : bool
#         If True, rescale dens so that the total number
#         ∫ dens d(measure) ≈ 1, computed using bin integrals:
#             total = Σ (dens * widths)
#             dens /= total

#     Returns
#     -------
#     centers : array
#         Bin centers in x.
#     dens : array
#         Number density per d(measure) (e.g., dN/dlnx).
#     edges : array
#         The input edges, passed through.
#     """
#     x = np.asarray(x)
#     w = np.asarray(weights, dtype=float)

#     # Histogram of counts/weights per bin
#     H, _ = np.histogram(x, bins=edges, weights=w)

#     # Widths in the chosen measure
#     widths = bin_widths(edges, measure)

#     # Density per d{measure}x on each bin (cell-average): dN/d(measure)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         dens = np.where(widths > 0, H / widths, 0.0)

#     # Bin centers in x
#     if measure == "linear":
#         centers = 0.5 * (edges[:-1] + edges[1:])
#     else:
#         centers = np.sqrt(edges[:-1] * edges[1:])

#     if normalize:
#         # Normalization is done via **bin integrals** in the chosen measure:
#         # N_total = Σ (dens * widths) = Σ H
#         total = (dens * widths).sum()
#         if total > 0:
#             dens = dens / total

#     return centers, dens, edges


# def density1d_cdf_map(
#     x_src_centers: np.ndarray,
#     dens_src: np.ndarray,
#     edges_tgt: np.ndarray,
#     measure: str = "ln",
# ):
#     """
#     Conservative mapping of a tabulated 1D **number density** (per d{measure}x)
#     onto target edges.

#     Steps:
#       - Build "source" edges around input centers.
#       - Integrate dens_src over source cells in u-space to get a CDF.
#       - Interpolate the CDF to target edges.
#       - Difference to recover counts per target bin.
#       - Divide by target widths to get number density on target bins.

#     Returns (centers_tgt, dens_tgt, edges_tgt).
#     """
#     x_src = np.asarray(x_src_centers)
#     y_src = np.asarray(dens_src, dtype=float)

#     # Build edges around centers
#     if measure == "ln":
#         r = np.sqrt(x_src[1:] / x_src[:-1])
#         src_edges = np.empty(x_src.size + 1)
#         src_edges[1:-1] = x_src[:-1] * r
#         src_edges[0] = x_src[0] / r[0]
#         src_edges[-1] = x_src[-1] * r[-1]
#     else:
#         d = 0.5 * (x_src[1:] - x_src[:-1])
#         src_edges = np.empty(x_src.size + 1)
#         src_edges[1:-1] = 0.5 * (x_src[:-1] + x_src[1:])
#         src_edges[0] = x_src[0] - d[0]
#         src_edges[-1] = x_src[-1] + d[-1]

#     # Integrate to CDF in u-space
#     du_src = np.diff(_u_from_x(src_edges, measure))
#     cell_N = y_src * du_src  # numbers per bin
#     N_src = np.concatenate([[0.0], np.cumsum(cell_N)])  # CDF at src_edges

#     u_edges_tgt = _u_from_x(edges_tgt, measure)
#     if _PCHIP is not None and N_src.size >= 2:
#         # monotone interpolator if available
#         N_of_u = _PCHIP(_u_from_x(src_edges, measure), N_src, extrapolate=True)
#         N_edges = N_of_u(u_edges_tgt)
#     else:
#         N_edges = np.interp(
#             u_edges_tgt,
#             _u_from_x(src_edges, measure),
#             N_src,
#             left=0.0,
#             right=N_src[-1],
#         )

#     widths = bin_widths(edges_tgt, measure)
#     dN = np.maximum(0.0, np.diff(N_edges))
#     with np.errstate(divide="ignore", invalid="ignore"):
#         dens_tgt = np.where(widths > 0, dN / widths, 0.0)

#     if measure == "linear":
#         centers = 0.5 * (edges_tgt[:-1] + edges_tgt[1:])
#     else:
#         centers = np.sqrt(edges_tgt[:-1] * edges_tgt[1:])

#     return centers, dens_tgt, edges_tgt


# def kde1d_in_measure(
#     x: np.ndarray,
#     weights: np.ndarray,
#     xq: np.ndarray,
#     measure: str = "ln",
#     normalize: bool = False,
# ):
#     """
#     Smooth estimate of **number density** wrt the chosen measure using a KDE in u-space.

#     Parameters
#     ----------
#     x : array
#         Sample locations.
#     weights : array
#         Sample weights.
#     xq : array
#         Query points in x.
#     measure : {"ln", "linear"}
#         - "ln": dens is dN/d(ln x)
#         - "linear": dens is dN/dx
#     normalize : bool
#         If True, dens is rescaled so that the integral over u (ln x or x)
#         approximated by trapz(dens, u) is 1.

#     Returns
#     -------
#     dens : array
#         KDE-evaluated number density per d(measure) at xq.
#     """
#     try:
#         from scipy.stats import gaussian_kde
#     except Exception as e:
#         raise RuntimeError("scipy is required for KDE") from e

#     x = np.asarray(x)
#     w = np.asarray(weights, dtype=float)
#     xq = np.asarray(xq)

#     # Work in u-space
#     u = _u_from_x(x, measure)
#     u_q = _u_from_x(xq, measure)

#     kde = gaussian_kde(u, weights=w)
#     dens = kde(u_q)  # dens(u) per du

#     if normalize:
#         total = np.trapz(dens, u_q)
#         if total > 0:
#             dens = dens / total

#     return dens


# # ---------- 2D distributions ----------

# def density2d_from_samples(
#     x: np.ndarray,
#     y: np.ndarray,
#     weights: np.ndarray,
#     edges_x: np.ndarray,
#     edges_y: np.ndarray,
#     measure_x: str = "ln",
#     measure_y: str = "ln",
#     normalize: bool = False,
# ):
#     """
#     Conservative 2D histogram -> **number density** per
#     d{measure_x}x d{measure_y}y.

#     Returns
#     -------
#     centers_x, centers_y : arrays
#         Bin centers in x and y.
#     dens : array
#         Number density per d(measure_x) d(measure_y).
#     edges_x, edges_y : arrays
#         The input edges, passed through.
#     """
#     H, ex, ey = np.histogram2d(
#         x,
#         y,
#         bins=[edges_x, edges_y],
#         weights=np.asarray(weights, dtype=float),
#     )

#     wx = bin_widths(ex, measure_x)[:, None]
#     wy = bin_widths(ey, measure_y)[None, :]

#     # dens = counts / (width_x * width_y) → dN/(dmeasure_x dmeasure_y)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         dens = np.where((wx > 0) & (wy > 0), H / (wx * wy), 0.0)

#     if normalize:
#         # Normalize via integrals in (u_x, u_y):
#         # N_total = Σ dens * wx * wy = Σ H
#         total = (dens * wx * wy).sum()
#         if total > 0:
#             dens = dens / total

#     if measure_x == "linear":
#         cx = 0.5 * (ex[:-1] + ex[1:])
#     else:
#         cx = np.sqrt(ex[:-1] * ex[1:])

#     if measure_y == "linear":
#         cy = 0.5 * (ey[:-1] + ey[1:])
#     else:
#         cy = np.sqrt(ey[:-1] * ey[1:])

#     return cx, cy, dens, ex, ey


# def density2d_cdf_map(
#     src_edges_x: np.ndarray,
#     src_edges_y: np.ndarray,
#     dens_src: np.ndarray,  # per d{measure_x}x d{measure_y}y on src cells
#     tgt_edges_x: np.ndarray,
#     tgt_edges_y: np.ndarray,
#     measure_x: str = "ln",
#     measure_y: str = "ln",
# ):
#     """
#     Conservative mapping of a 2D **number density** on a rectilinear source grid
#     onto target edges. Uses the 2D CDF in (u_x, u_y), then inclusion-exclusion
#     per target cell.
#     """
#     if _RGI is None:
#         raise RuntimeError("scipy RegularGridInterpolator is required for 2D CDF mapping")

#     ux_e_src = _u_from_x(src_edges_x, measure_x)
#     uy_e_src = _u_from_x(src_edges_y, measure_y)
#     dux = np.diff(ux_e_src)
#     duy = np.diff(uy_e_src)

#     # integrate density over source cells to get counts
#     cell_N = dens_src * (dux[:, None] * duy[None, :])

#     # Build CDF on edge grid (nx+1, ny+1)
#     N = np.zeros((cell_N.shape[0] + 1, cell_N.shape[1] + 1))
#     N[1:, 1:] = cell_N.cumsum(axis=0).cumsum(axis=1)

#     # Interpolate CDF to target edge grid
#     ux_e_tgt = _u_from_x(tgt_edges_x, measure_x)
#     uy_e_tgt = _u_from_x(tgt_edges_y, measure_y)
#     rgi = _RGI((ux_e_src, uy_e_src), N, bounds_error=False, fill_value=(N[-1, -1]))
#     Ux, Uy = np.meshgrid(ux_e_tgt, uy_e_tgt, indexing="ij")
#     Nt = rgi(np.stack([Ux, Uy], axis=-1))  # shape (Nx+1, Ny+1)

#     # Inclusion-exclusion to recover counts per target cell
#     dN = Nt[1:, 1:] - Nt[:-1, 1:] - Nt[1:, :-1] + Nt[:-1, :-1]
#     dux_t = np.diff(ux_e_tgt)[:, None]
#     duy_t = np.diff(uy_e_tgt)[None, :]
#     with np.errstate(divide="ignore", invalid="ignore"):
#         dens_tgt = np.where((dux_t > 0) & (duy_t > 0), dN / (dux_t * duy_t), 0.0)

#     if measure_x == "linear":
#         cx = 0.5 * (tgt_edges_x[:-1] + tgt_edges_x[1:])
#     else:
#         cx = np.sqrt(tgt_edges_x[:-1] * tgt_edges_x[1:])

#     if measure_y == "linear":
#         cy = 0.5 * (tgt_edges_y[:-1] + tgt_edges_y[1:])
#     else:
#         cy = np.sqrt(tgt_edges_y[:-1] * tgt_edges_y[1:])

#     return cx, cy, dens_tgt, tgt_edges_x, tgt_edges_y
