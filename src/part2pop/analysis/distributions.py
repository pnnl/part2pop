from __future__ import annotations

import warnings

import numpy as np

_TRAPEZOID_FUNC = getattr(np, "trapezoid", None)
_TRAPZ_FUNC = getattr(np, "trapz", None)

if _TRAPZ_FUNC is None and _TRAPEZOID_FUNC is not None:
    _TRAPZ_FUNC = _TRAPEZOID_FUNC
    np.trapz = _TRAPZ_FUNC

if _TRAPEZOID_FUNC is None and _TRAPZ_FUNC is not None:
    _TRAPEZOID_FUNC = _TRAPZ_FUNC
    np.trapezoid = _TRAPEZOID_FUNC

def _trapezoid_integrate(y, x=None, dx=1.0, axis=-1):
    """
    Call the available NumPy trapezoidal integration routine.

    NumPy 2.0 dropped `np.trapz` in favor of `np.trapezoid`, while older
    releases only expose `np.trapz`.  This helper always calls whichever
    function exists so the rest of the module can rely on a single name.
    """
    if _TRAPEZOID_FUNC is not None:
        return _TRAPEZOID_FUNC(y, x=x, dx=dx, axis=axis)
    if _TRAPZ_FUNC is not None:
        return _TRAPZ_FUNC(y, x=x, dx=dx, axis=axis)
    raise AttributeError("NumPy installation lacks trapezoid/trapz integrators")
    # func = getattr(np, "trapezoid", None)
    # if func is None:
    #     func = getattr(np, "trapz", None)
    # if func is None:
    #     raise AttributeError("NumPy installation lacks trapezoid/trapz integrators")
    # return func(y, x=x, dx=dx, axis=axis)

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
        total = _trapezoid_integrate(dens, u_centers)
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


def _kde_integration_widths(query):
    """Validate the quadrature grid used by normalization and point masses."""
    with np.errstate(over="ignore", invalid="ignore"):
        widths = np.diff(query)
    if (query.size < 2 or not np.all(np.isfinite(widths))
            or np.any(widths <= 0)):
        raise ValueError("KDE integration needs at least two increasing nodes with finite widths")
    return widths


def _point_masses_density(locations, weights, query, bounds=None):
    """Deposit weighted point masses on adjacent query nodes (linear binning).

    Each point's weight is split between the two nodes around it in
    proportion to proximity, then divided by those nodes' trapezoidal
    quadrature weights. ``trapz(density, query)`` therefore equals the total
    weight of the points inside ``bounds``, and interior first moments are
    conserved. A point between an outer bound and the nearest node is placed
    at that node (count conserved, first moment shifted). Points outside
    ``bounds`` are dropped.

    Returns ``(density, dropped_weight)``.
    """
    widths = _kde_integration_widths(query)
    if bounds is None:
        bounds = query[[0, -1]]
    elif (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] > query[0] or bounds[1] < query[-1]):
        raise ValueError("Point-mass bounds must be finite and enclose the query grid")
    locations = np.atleast_1d(np.asarray(locations, dtype=float))
    weights = np.broadcast_to(np.asarray(weights, dtype=float), locations.shape)
    inside = (locations >= bounds[0]) & (locations <= bounds[1])
    dropped = float(np.sum(weights[~inside]))
    # A point within an outer half-bin is represented at its nearest endpoint.
    kept = np.clip(locations[inside], query[0], query[-1])
    kept_weights = weights[inside]
    quadrature_weights = np.empty_like(query, dtype=float)
    quadrature_weights[0] = widths[0] / 2
    quadrature_weights[-1] = widths[-1] / 2
    quadrature_weights[1:-1] = widths[:-1] / 2 + widths[1:] / 2
    # Clipping guarantees a right-hand index of at least one.
    right = np.minimum(np.searchsorted(query, kept, side="right"), query.size - 1)
    left = right - 1
    fraction = (kept - query[left]) / widths[left]
    mass = np.zeros_like(query, dtype=float)
    np.add.at(mass, left, kept_weights * (1 - fraction))
    np.add.at(mass, right, kept_weights * fraction)
    return mass / quadrature_weights, dropped


def _point_mass_density(location, total, query, bounds=None):
    """Deposit one point mass; see :func:`_point_masses_density`."""
    density, _ = _point_masses_density([location], [total], query, bounds)
    return density


# A Gaussian sampled at uniform spacing d integrates (trapezoid rule) with
# relative error of about 2*exp(-2*pi**2*(h/d)**2): 3e-4 at h = 2d/3, but 1.4%
# at h = d/2 and 34% at h = 0.3d. Narrower kernels are deposited instead.
_MIN_RESOLVED_BANDWIDTH = 2.0 / 3.0


def _kernel_is_unresolved(bandwidth, query):
    """Return True when a Gaussian kernel is too narrow for the query spacing.

    Sampling such a kernel at the query nodes cannot represent its integral:
    a node next to a sample sees the peak, and a sample between nodes is
    missed. A kernel at least two thirds of the coarsest spacing integrates to
    within about 3e-4 when sampled. Only increasing grids of two or more nodes
    qualify; other query grids are sampled directly.
    """
    if query.size < 2:
        return False
    widths = np.diff(query)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        return False
    return bandwidth < _MIN_RESOLVED_BANDWIDTH * np.max(widths)


def kde1d_in_measure(
    x: np.ndarray,
    weights: np.ndarray,
    xq: np.ndarray,
    measure: str = "ln",
    normalize: bool = False,
    point_mass_bounds=None,
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
    point_mass_bounds : pair of floats, optional
        Outer bin edges enclosing xq, used when samples are deposited as point
        masses (see Notes). Points within these bounds but beyond the query
        nodes are deposited at the nearest endpoint. Without bounds, support
        is the query interval.

    Returns
    -------
    dens : array
        KDE-evaluated number density per d(measure) at xq.

    Notes
    -----
    When all samples share one value, or the fitted Gaussian bandwidth is
    below two thirds of the largest spacing of the (sorted) query grid,
    sampling the kernel at the nodes cannot represent its integral. Samples are then
    deposited as point masses on their two neighbouring nodes (linear
    binning): the trapezoidal integral over the query coordinates equals the
    total weight inside the support, and interior first moments are kept.
    Weight outside the support is dropped with a RuntimeWarning. Broader
    kernels are sampled directly, as before.
    """
    try:
        from scipy.stats import gaussian_kde
    except Exception as e:
        raise RuntimeError("scipy is required for KDE") from e

    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    xq = np.asarray(xq, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or xq.ndim != 1:
        raise ValueError("KDE samples/weights must match 1D arrays; query must be 1D")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(xq)):
        raise ValueError("KDE sample and query coordinates must be finite")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("KDE weights must be finite and nonnegative")
    u_q = _u_from_x(xq, measure)
    if normalize:
        _kde_integration_widths(u_q)
    positive = w > 0
    if not np.any(positive):
        return np.zeros_like(xq)
    u = _u_from_x(x[positive], measure)
    w = w[positive]
    total_weight = np.sum(w)
    if not np.isfinite(total_weight):
        raise ValueError("Total KDE weight must be finite")

    # Decide and deposit on the nodes in increasing order, so a raw KDE gives
    # the same values whatever order the query nodes are supplied in.
    order = np.argsort(u_q, kind="stable")
    u_sorted = u_q[order]

    # Weighted covariance divides by 1 - sum(p**2). A dominant weight can
    # round this denominator to zero even with distinct samples. SciPy then
    # raises ValueError for a nonfinite covariance, not LinAlgError. Detect
    # that numerical degeneracy without swallowing unrelated ValueErrors.
    scaled_weights = w / np.max(w)
    probabilities = scaled_weights / np.sum(scaled_weights)
    covariance_dof = 1.0 - np.sum(probabilities**2)
    kde = None
    if not np.all(u == u[0]) and covariance_dof > 8 * np.finfo(float).eps:
        try:
            kde = gaussian_kde(u, weights=probabilities)
        except np.linalg.LinAlgError:
            kde = None  # numerically singular spread: treat as point masses
    if kde is None or _kernel_is_unresolved(float(np.sqrt(kde.covariance[0, 0])), u_sorted):
        bounds = None if point_mass_bounds is None else _u_from_x(
            np.asarray(point_mass_bounds, dtype=float), measure
        )
        # Unit total for a normalized estimate avoids scaling extreme weights.
        deposit = w / total_weight if normalize else w
        dens_sorted, dropped = _point_masses_density(u, deposit, u_sorted, bounds)
        dens = np.empty_like(dens_sorted)
        dens[order] = dens_sorted
        if dropped > 0:
            share = dropped / (1.0 if normalize else total_weight)
            warnings.warn(
                f"{share:.3g} of the KDE weight lies outside the distribution "
                "support and is not represented.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        # scipy normalizes weights: restore counts only for number density.
        dens = kde(u_q)
        if not normalize:
            dens = dens * total_weight

    if normalize:
        total = _trapezoid_integrate(dens, u_q)
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
        tmp = _trapezoid_integrate(dens, uy, axis=1)
        total = _trapezoid_integrate(tmp, ux)
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
