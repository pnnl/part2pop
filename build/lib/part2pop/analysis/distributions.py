from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    from scipy.interpolate import RegularGridInterpolator as _RGI
except Exception:
    _PCHIP = None
    _RGI = None


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def make_edges(xmin: float, xmax: float, n_bins: int, scale: str = "log"):
    """
    Return (edges, centers) on a linear or logarithmic scale.

    Parameters
    ----------
    xmin, xmax : float
        Range of x (must be > 0 for log scale).
    n_bins : int
        Number of bins.
    scale : {"log", "linear"}
        Type of spacing for the bins.

    Returns
    -------
    edges : ndarray, shape (n_bins+1,)
        Bin edges in x.
    centers : ndarray, shape (n_bins,)
        Bin centers in x.

    Notes
    -----
    For 'log' scale, edges are spaced logarithmically in x and centers
    are geometric means of neighboring edges.
    """
    if scale not in ("log", "linear"):
        raise ValueError("scale must be 'log' or 'linear'")

    if scale == "log":
        if xmin <= 0 or xmax <= 0:
            raise ValueError("xmin, xmax must be positive for log scale")
        edges = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    return edges, centers


def bin_widths(edges: np.ndarray, measure: str = "ln") -> np.ndarray:
    """
    Return bin widths in the chosen measure.

    Parameters
    ----------
    edges : ndarray
        Bin edges in x (must be > 0 for measure="ln").
    measure : {"ln", "linear"}
        Measure in which the density is defined:
          - "ln":     widths = Δln(x)
          - "linear": widths = Δx

    Returns
    -------
    widths : ndarray, shape (len(edges)-1,)
        Bin widths in the chosen measure.
    """
    edges = np.asarray(edges, dtype=float)
    if measure == "ln":
        if np.any(edges <= 0.0):
            raise ValueError("edges must be positive for measure='ln'")
        return np.diff(np.log(edges))
    elif measure == "linear":
        return np.diff(edges)
    else:
        raise ValueError("measure must be 'ln' or 'linear'")


def _u_from_x(x: np.ndarray, measure: str) -> np.ndarray:
    """
    Change of variable x -> u for integration.

    Parameters
    ----------
    x : ndarray
    measure : {"ln", "linear"}

    Returns
    -------
    u : ndarray
        Integration coordinate:
          - u = ln(x)     if measure == "ln"
          - u = x         if measure == "linear"
    """
    if measure == "ln":
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0.0):
            raise ValueError("x must be positive for measure='ln'")
        return np.log(x)
    elif measure == "linear":
        return np.asarray(x, dtype=float)
    else:
        raise ValueError("measure must be 'ln' or 'linear'")


# ---------------------------------------------------------------------------
# 1D distributions
# ---------------------------------------------------------------------------

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
    edges = np.asarray(edges, dtype=float)

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
        # dens remains a number density (H / widths).
        # We normalize using the same quadrature that users/tests use:
        #   total = ∫ dens d(measure) ≈ trapz(dens, u_centers)
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

    Parameters
    ----------
    x_src_centers : array
        Source bin centers in x.
    dens_src : array
        Source number density per d(measure), defined at x_src_centers.
    edges_tgt : array
        Target bin edges in x.
    measure : {"ln", "linear"}
        Measure in which dens_src is defined.

    Returns
    -------
    x_tgt_centers : array
        Target bin centers in x.
    dens_tgt : array
        Target number density per d(measure), conservative mapping of dens_src.
    edges_tgt : array
        The input target edges, passed through.

    Notes
    -----
    This operates in the integration coordinate u (ln x or x), builds
    a CDF, samples it at target edges, and differences it to get the
    new density, ensuring conservation of total ∫ dens d(measure).
    """
    x_src_centers = np.asarray(x_src_centers, dtype=float)
    dens_src = np.asarray(dens_src, dtype=float)
    edges_tgt = np.asarray(edges_tgt, dtype=float)

    if x_src_centers.ndim != 1:
        raise ValueError("x_src_centers must be 1D")
    if dens_src.shape != x_src_centers.shape:
        raise ValueError("dens_src must have same shape as x_src_centers")
    if edges_tgt.ndim != 1 or edges_tgt.size < 2:
        raise ValueError("edges_tgt must be 1D with at least 2 elements")

    # Work in integration coordinate u
    u_src_centers = _u_from_x(x_src_centers, measure)
    u_tgt_edges = _u_from_x(edges_tgt, measure)

    # Build source edges in u around the centers
    du_src_between = np.diff(u_src_centers)
    if du_src_between.size == 0:
        # Single-bin degenerate case: pick an arbitrary small width
        du = 1.0
        u_src_edges = np.array([u_src_centers[0] - 0.5 * du,
                                u_src_centers[0] + 0.5 * du])
    else:
        du_left = du_src_between[0]
        du_right = du_src_between[-1]
        u_edges_inner = 0.5 * (u_src_centers[1:] + u_src_centers[:-1])
        u_first = u_src_centers[0] - 0.5 * du_left
        u_last = u_src_centers[-1] + 0.5 * du_right
        u_src_edges = np.concatenate([[u_first], u_edges_inner, [u_last]])

    # Total in each source bin
    du_src = np.diff(u_src_edges)
    N_src = dens_src * du_src

    # CDF at source edges
    cdf_src_edges = np.concatenate([[0.0], np.cumsum(N_src)])
    total_src = cdf_src_edges[-1]

    # Interpolate CDF to target edges in u
    cdf_tgt_edges = np.interp(
        u_tgt_edges,
        u_src_edges,
        cdf_src_edges,
        left=0.0,
        right=total_src,
    )

    # Bin totals in target bins
    N_tgt = np.diff(cdf_tgt_edges)

    # Convert back to density in measure units
    du_tgt = np.diff(u_tgt_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_tgt = np.where(du_tgt > 0, N_tgt / du_tgt, 0.0)

    # Target centers in x
    if measure == "linear":
        x_tgt_centers = 0.5 * (edges_tgt[:-1] + edges_tgt[1:])
    else:
        x_tgt_centers = np.sqrt(edges_tgt[:-1] * edges_tgt[1:])

    return x_tgt_centers, dens_tgt, edges_tgt


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
        Query points where the density should be evaluated.
    measure : {"ln", "linear"}
        Measure in which the density is defined:
          - "ln": dens ~ dN/d(ln x)
          - "linear": dens ~ dN/dx
    normalize : bool
        If True, rescale dens so that the integral in u-space
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


# ---------------------------------------------------------------------------
# 2D distributions
# ---------------------------------------------------------------------------

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
    Conservative 2D histogram of samples into a **number density** wrt chosen measures.

    Parameters
    ----------
    x, y : array
        Sample locations (e.g., diameters and another variable).
    weights : array
        Sample weights (e.g., counts or number concentration per sample).
    edges_x, edges_y : array
        Bin edges in x and y.
    measure_x, measure_y : {"ln", "linear"}
        Measures in which the density is defined:
          - "ln":     dens is dN/d(ln x) (or y)
          - "linear": dens is dN/dx      (or dy)
    normalize : bool
        If True, rescale dens so that the integral over both measures
        approximated by a double integral is 1.

    Returns
    -------
    centers_x, centers_y : array
        Bin centers in x and y.
    dens : array, shape (nx, ny)
        Number density per d(measure_x) d(measure_y).
    edges_x, edges_y : array
        The input edges, passed through.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(weights, dtype=float)
    edges_x = np.asarray(edges_x, dtype=float)
    edges_y = np.asarray(edges_y, dtype=float)

    H, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=w)

    widths_x = bin_widths(edges_x, measure_x)  # Δux (ln x or x)
    widths_y = bin_widths(edges_y, measure_y)  # Δuy (ln y or y)

    # Broadcast widths to 2D grid
    wx = widths_x[:, None]
    wy = widths_y[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where((wx > 0) & (wy > 0), H / (wx * wy), 0.0)

    # Centers
    if measure_x == "linear":
        cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    else:
        cx = np.sqrt(edges_x[:-1] * edges_x[1:])

    if measure_y == "linear":
        cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    else:
        cy = np.sqrt(edges_y[:-1] * edges_y[1:])

    if normalize:
        ux = _u_from_x(cx, measure_x)
        uy = _u_from_x(cy, measure_y)
        # Approximate double integral via tensor product trapz
        # integrate over y first, then x
        tmp = np.trapz(dens, uy, axis=1)
        total = np.trapz(tmp, ux)
        if total > 0:
            dens = dens / total

    return cx, cy, dens, edges_x, edges_y


def density2d_cdf_map(
    x_src_centers: np.ndarray,
    y_src_centers: np.ndarray,
    dens_src: np.ndarray,
    edges_x_tgt: np.ndarray,
    edges_y_tgt: np.ndarray,
    measure_x: str = "ln",
    measure_y: str = "ln",
):
    """
    (Advanced) Conservative mapping of a tabulated 2D **number density** onto
    target edges.

    Parameters
    ----------
    x_src_centers, y_src_centers : array
        Source bin centers in x and y.
    dens_src : array, shape (nx, ny)
        Source number density per d(measure_x) d(measure_y).
    edges_x_tgt, edges_y_tgt : array
        Target bin edges in x and y.
    measure_x, measure_y : {"ln", "linear"}
        Measures for x and y.

    Returns
    -------
    cx_tgt, cy_tgt : array
        Target bin centers in x and y.
    dens_tgt : array, shape (nx_tgt, ny_tgt)
        Target number density per d(measure_x) d(measure_y).
    edges_x_tgt, edges_y_tgt : array
        The input target edges, passed through.

    Notes
    -----
    This is a nontrivial operation if done fully CDF-based in 2D.
    Here we implement a separable, conservative mapping:

        1) Map in x direction for each fixed y-bin using 1D CDF mapping.
        2) Then map in y direction for each fixed x-bin using 1D CDF mapping.

    This preserves the total ∫ dens d(measure_x) d(measure_y), but exact
    2D CDF shape is only approximated. For many practical cases this is
    adequate. If a fully 2D CDF-based mapping is required, a dedicated
    implementation should replace this.
    """
    x_src_centers = np.asarray(x_src_centers, dtype=float)
    y_src_centers = np.asarray(y_src_centers, dtype=float)
    dens_src = np.asarray(dens_src, dtype=float)
    edges_x_tgt = np.asarray(edges_x_tgt, dtype=float)
    edges_y_tgt = np.asarray(edges_y_tgt, dtype=float)

    if dens_src.shape != (x_src_centers.size, y_src_centers.size):
        raise ValueError("dens_src must have shape (nx, ny) matching centers")

    # Step 1: map along x for each fixed y-bin
    nx_tgt = edges_x_tgt.size - 1
    ny_src = y_src_centers.size
    dens_xmapped = np.zeros((nx_tgt, ny_src), dtype=float)
    x_tgt_centers = None

    for j in range(ny_src):
        cx_tgt, dens_x, _ = density1d_cdf_map(
            x_src_centers=x_src_centers,
            dens_src=dens_src[:, j],
            edges_tgt=edges_x_tgt,
            measure=measure_x,
        )
        dens_xmapped[:, j] = dens_x
        if x_tgt_centers is None:
            x_tgt_centers = cx_tgt

    # Step 2: map along y for each fixed x-bin
    ny_tgt = edges_y_tgt.size - 1
    dens_tgt = np.zeros((nx_tgt, ny_tgt), dtype=float)
    y_tgt_centers = None

    for i in range(nx_tgt):
        cy_tgt, dens_y, _ = density1d_cdf_map(
            x_src_centers=y_src_centers,
            dens_src=dens_xmapped[i, :],
            edges_tgt=edges_y_tgt,
            measure=measure_y,
        )
        dens_tgt[i, :] = dens_y
        if y_tgt_centers is None:
            y_tgt_centers = cy_tgt

    return x_tgt_centers, y_tgt_centers, dens_tgt, edges_x_tgt, edges_y_tgt



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

#     - measure == "ln":     widths = d(ln x) between edges
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
#           - "ln":     dens is dN/d(ln x)
#           - "linear": dens is dN/dx
#     normalize : bool
#         If True, rescale dens so that the integral
#             ∫ dens d(measure) ≈ 1
#         when computed via trapz over centers in u-space:
#             u = ln(centers) if measure == "ln"
#             u = centers     if measure == "linear"

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
#         # IMPORTANT:
#         #  - dens remains a number density (H / widths).
#         #  - We normalize using the same quadrature that users/tests use:
#         #      total = ∫ dens d(measure) ≈ trapz(dens, u_centers)
#         #    where u_centers is ln(centers) or centers depending on measure.
#         u_centers = _u_from_x(centers, measure)
#         total = np.trapz(dens, u_centers)
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
#     u_edges_src = _u_from_x(src_edges, measure)
#     du_src = np.diff(u_edges_src)
#     cell_N = y_src * du_src  # numbers per bin
#     N_src = np.concatenate([[0.0], np.cumsum(cell_N)])  # CDF at src_edges

#     u_edges_tgt = _u_from_x(edges_tgt, measure)
#     if _PCHIP is not None and N_src.size >= 2:
#         # monotone interpolator if available
#         N_of_u = _PCHIP(u_edges_src, N_src, extrapolate=True)
#         N_edges = N_of_u(u_edges_tgt)
#     else:
#         N_edges = np.interp(
#             u_edges_tgt,
#             u_edges_src,
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
#         - "ln":     dens is dN/d(ln x)
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
#         # N_total = ∫∫ dens d(measure_x)d(measure_y) ≈ Σ dens * wx * wy
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

# # from __future__ import annotations
# # import numpy as np

# # try:
# #     from scipy.interpolate import PchipInterpolator as _PCHIP
# #     from scipy.interpolate import RegularGridInterpolator as _RGI
# # except Exception:
# #     _PCHIP = None
# #     _RGI = None


# # # ---------- Grid helpers ----------

# # def make_edges(xmin: float, xmax: float, n_bins: int, scale: str = "log"):
# #     """Return (edges, centers) on linear or logarithmic scale."""
# #     if scale == "log":
# #         edges = np.geomspace(xmin, xmax, n_bins + 1)
# #         centers = np.sqrt(edges[:-1] * edges[1:])
# #     elif scale == "linear":
# #         edges = np.linspace(xmin, xmax, n_bins + 1)
# #         centers = 0.5 * (edges[:-1] + edges[1:])
# #     else:
# #         raise ValueError("scale must be 'log' or 'linear'")
# #     return edges, centers


# # def bin_widths(edges: np.ndarray, measure: str = "ln"):
# #     """
# #     Widths in the measure of integration.

# #     - measure == "ln":   widths = d(ln x) between edges
# #     - measure == "linear": widths = dx between edges
# #     """
# #     if measure == "ln":
# #         return np.log(edges[1:]) - np.log(edges[:-1])
# #     elif measure == "linear":
# #         return edges[1:] - edges[:-1]
# #     else:
# #         raise ValueError("measure must be 'ln' or 'linear'")


# # def _u_from_x(x: np.ndarray, measure: str):
# #     """Change of variable to the integration coordinate u."""
# #     if measure == "ln":
# #         return np.log(x)
# #     elif measure == "linear":
# #         return np.asarray(x)
# #     else:
# #         raise ValueError("measure must be 'ln' or 'linear'")


# # # ---------- 1D distributions ----------

# # def density1d_from_samples(
# #     x: np.ndarray,
# #     weights: np.ndarray,
# #     edges: np.ndarray,
# #     measure: str = "ln",
# #     normalize: bool = False,
# # ):
# #     """
# #     Conservative histogram of samples into a **number density** wrt the chosen measure.

# #     Parameters
# #     ----------
# #     x : array
# #         Sample locations (e.g., diameters).
# #     weights : array
# #         Sample weights (e.g., counts or number concentration per sample).
# #     edges : array
# #         Bin edges in x.
# #     measure : {"ln", "linear"}
# #         Measure in which the density is defined:
# #           - "ln":    dens is dN/d(ln x)
# #           - "linear": dens is dN/dx
# #     normalize : bool
# #         If True, rescale dens so that the total number
# #         ∫ dens d(measure) ≈ 1, computed using bin integrals:
# #             total = Σ (dens * widths)
# #             dens /= total

# #     Returns
# #     -------
# #     centers : array
# #         Bin centers in x.
# #     dens : array
# #         Number density per d(measure) (e.g., dN/dlnx).
# #     edges : array
# #         The input edges, passed through.
# #     """
# #     x = np.asarray(x)
# #     w = np.asarray(weights, dtype=float)

# #     # Histogram of counts/weights per bin
# #     H, _ = np.histogram(x, bins=edges, weights=w)

# #     # Widths in the chosen measure
# #     widths = bin_widths(edges, measure)

# #     # Density per d{measure}x on each bin (cell-average): dN/d(measure)
# #     with np.errstate(divide="ignore", invalid="ignore"):
# #         dens = np.where(widths > 0, H / widths, 0.0)

# #     # Bin centers in x
# #     if measure == "linear":
# #         centers = 0.5 * (edges[:-1] + edges[1:])
# #     else:
# #         centers = np.sqrt(edges[:-1] * edges[1:])

# #     if normalize:
# #         # Normalization is done via **bin integrals** in the chosen measure:
# #         # N_total = Σ (dens * widths) = Σ H
# #         total = (dens * widths).sum()
# #         if total > 0:
# #             dens = dens / total

# #     return centers, dens, edges


# # def density1d_cdf_map(
# #     x_src_centers: np.ndarray,
# #     dens_src: np.ndarray,
# #     edges_tgt: np.ndarray,
# #     measure: str = "ln",
# # ):
# #     """
# #     Conservative mapping of a tabulated 1D **number density** (per d{measure}x)
# #     onto target edges.

# #     Steps:
# #       - Build "source" edges around input centers.
# #       - Integrate dens_src over source cells in u-space to get a CDF.
# #       - Interpolate the CDF to target edges.
# #       - Difference to recover counts per target bin.
# #       - Divide by target widths to get number density on target bins.

# #     Returns (centers_tgt, dens_tgt, edges_tgt).
# #     """
# #     x_src = np.asarray(x_src_centers)
# #     y_src = np.asarray(dens_src, dtype=float)

# #     # Build edges around centers
# #     if measure == "ln":
# #         r = np.sqrt(x_src[1:] / x_src[:-1])
# #         src_edges = np.empty(x_src.size + 1)
# #         src_edges[1:-1] = x_src[:-1] * r
# #         src_edges[0] = x_src[0] / r[0]
# #         src_edges[-1] = x_src[-1] * r[-1]
# #     else:
# #         d = 0.5 * (x_src[1:] - x_src[:-1])
# #         src_edges = np.empty(x_src.size + 1)
# #         src_edges[1:-1] = 0.5 * (x_src[:-1] + x_src[1:])
# #         src_edges[0] = x_src[0] - d[0]
# #         src_edges[-1] = x_src[-1] + d[-1]

# #     # Integrate to CDF in u-space
# #     du_src = np.diff(_u_from_x(src_edges, measure))
# #     cell_N = y_src * du_src  # numbers per bin
# #     N_src = np.concatenate([[0.0], np.cumsum(cell_N)])  # CDF at src_edges

# #     u_edges_tgt = _u_from_x(edges_tgt, measure)
# #     if _PCHIP is not None and N_src.size >= 2:
# #         # monotone interpolator if available
# #         N_of_u = _PCHIP(_u_from_x(src_edges, measure), N_src, extrapolate=True)
# #         N_edges = N_of_u(u_edges_tgt)
# #     else:
# #         N_edges = np.interp(
# #             u_edges_tgt,
# #             _u_from_x(src_edges, measure),
# #             N_src,
# #             left=0.0,
# #             right=N_src[-1],
# #         )

# #     widths = bin_widths(edges_tgt, measure)
# #     dN = np.maximum(0.0, np.diff(N_edges))
# #     with np.errstate(divide="ignore", invalid="ignore"):
# #         dens_tgt = np.where(widths > 0, dN / widths, 0.0)

# #     if measure == "linear":
# #         centers = 0.5 * (edges_tgt[:-1] + edges_tgt[1:])
# #     else:
# #         centers = np.sqrt(edges_tgt[:-1] * edges_tgt[1:])

# #     return centers, dens_tgt, edges_tgt


# # def kde1d_in_measure(
# #     x: np.ndarray,
# #     weights: np.ndarray,
# #     xq: np.ndarray,
# #     measure: str = "ln",
# #     normalize: bool = False,
# # ):
# #     """
# #     Smooth estimate of **number density** wrt the chosen measure using a KDE in u-space.

# #     Parameters
# #     ----------
# #     x : array
# #         Sample locations.
# #     weights : array
# #         Sample weights.
# #     xq : array
# #         Query points in x.
# #     measure : {"ln", "linear"}
# #         - "ln": dens is dN/d(ln x)
# #         - "linear": dens is dN/dx
# #     normalize : bool
# #         If True, dens is rescaled so that the integral over u (ln x or x)
# #         approximated by trapz(dens, u) is 1.

# #     Returns
# #     -------
# #     dens : array
# #         KDE-evaluated number density per d(measure) at xq.
# #     """
# #     try:
# #         from scipy.stats import gaussian_kde
# #     except Exception as e:
# #         raise RuntimeError("scipy is required for KDE") from e

# #     x = np.asarray(x)
# #     w = np.asarray(weights, dtype=float)
# #     xq = np.asarray(xq)

# #     # Work in u-space
# #     u = _u_from_x(x, measure)
# #     u_q = _u_from_x(xq, measure)

# #     kde = gaussian_kde(u, weights=w)
# #     dens = kde(u_q)  # dens(u) per du

# #     if normalize:
# #         total = np.trapz(dens, u_q)
# #         if total > 0:
# #             dens = dens / total

# #     return dens


# # # ---------- 2D distributions ----------

# # def density2d_from_samples(
# #     x: np.ndarray,
# #     y: np.ndarray,
# #     weights: np.ndarray,
# #     edges_x: np.ndarray,
# #     edges_y: np.ndarray,
# #     measure_x: str = "ln",
# #     measure_y: str = "ln",
# #     normalize: bool = False,
# # ):
# #     """
# #     Conservative 2D histogram -> **number density** per
# #     d{measure_x}x d{measure_y}y.

# #     Returns
# #     -------
# #     centers_x, centers_y : arrays
# #         Bin centers in x and y.
# #     dens : array
# #         Number density per d(measure_x) d(measure_y).
# #     edges_x, edges_y : arrays
# #         The input edges, passed through.
# #     """
# #     H, ex, ey = np.histogram2d(
# #         x,
# #         y,
# #         bins=[edges_x, edges_y],
# #         weights=np.asarray(weights, dtype=float),
# #     )

# #     wx = bin_widths(ex, measure_x)[:, None]
# #     wy = bin_widths(ey, measure_y)[None, :]

# #     # dens = counts / (width_x * width_y) → dN/(dmeasure_x dmeasure_y)
# #     with np.errstate(divide="ignore", invalid="ignore"):
# #         dens = np.where((wx > 0) & (wy > 0), H / (wx * wy), 0.0)

# #     if normalize:
# #         # Normalize via integrals in (u_x, u_y):
# #         # N_total = Σ dens * wx * wy = Σ H
# #         total = (dens * wx * wy).sum()
# #         if total > 0:
# #             dens = dens / total

# #     if measure_x == "linear":
# #         cx = 0.5 * (ex[:-1] + ex[1:])
# #     else:
# #         cx = np.sqrt(ex[:-1] * ex[1:])

# #     if measure_y == "linear":
# #         cy = 0.5 * (ey[:-1] + ey[1:])
# #     else:
# #         cy = np.sqrt(ey[:-1] * ey[1:])

# #     return cx, cy, dens, ex, ey


# # def density2d_cdf_map(
# #     src_edges_x: np.ndarray,
# #     src_edges_y: np.ndarray,
# #     dens_src: np.ndarray,  # per d{measure_x}x d{measure_y}y on src cells
# #     tgt_edges_x: np.ndarray,
# #     tgt_edges_y: np.ndarray,
# #     measure_x: str = "ln",
# #     measure_y: str = "ln",
# # ):
# #     """
# #     Conservative mapping of a 2D **number density** on a rectilinear source grid
# #     onto target edges. Uses the 2D CDF in (u_x, u_y), then inclusion-exclusion
# #     per target cell.
# #     """
# #     if _RGI is None:
# #         raise RuntimeError("scipy RegularGridInterpolator is required for 2D CDF mapping")

# #     ux_e_src = _u_from_x(src_edges_x, measure_x)
# #     uy_e_src = _u_from_x(src_edges_y, measure_y)
# #     dux = np.diff(ux_e_src)
# #     duy = np.diff(uy_e_src)

# #     # integrate density over source cells to get counts
# #     cell_N = dens_src * (dux[:, None] * duy[None, :])

# #     # Build CDF on edge grid (nx+1, ny+1)
# #     N = np.zeros((cell_N.shape[0] + 1, cell_N.shape[1] + 1))
# #     N[1:, 1:] = cell_N.cumsum(axis=0).cumsum(axis=1)

# #     # Interpolate CDF to target edge grid
# #     ux_e_tgt = _u_from_x(tgt_edges_x, measure_x)
# #     uy_e_tgt = _u_from_x(tgt_edges_y, measure_y)
# #     rgi = _RGI((ux_e_src, uy_e_src), N, bounds_error=False, fill_value=(N[-1, -1]))
# #     Ux, Uy = np.meshgrid(ux_e_tgt, uy_e_tgt, indexing="ij")
# #     Nt = rgi(np.stack([Ux, Uy], axis=-1))  # shape (Nx+1, Ny+1)

# #     # Inclusion-exclusion to recover counts per target cell
# #     dN = Nt[1:, 1:] - Nt[:-1, 1:] - Nt[1:, :-1] + Nt[:-1, :-1]
# #     dux_t = np.diff(ux_e_tgt)[:, None]
# #     duy_t = np.diff(uy_e_tgt)[None, :]
# #     with np.errstate(divide="ignore", invalid="ignore"):
# #         dens_tgt = np.where((dux_t > 0) & (duy_t > 0), dN / (dux_t * duy_t), 0.0)

# #     if measure_x == "linear":
# #         cx = 0.5 * (tgt_edges_x[:-1] + tgt_edges_x[1:])
# #     else:
# #         cx = np.sqrt(tgt_edges_x[:-1] * tgt_edges_x[1:])

# #     if measure_y == "linear":
# #         cy = 0.5 * (tgt_edges_y[:-1] + tgt_edges_y[1:])
# #     else:
# #         cy = np.sqrt(tgt_edges_y[:-1] * tgt_edges_y[1:])

# #     return cx, cy, dens_tgt, tgt_edges_x, tgt_edges_y
