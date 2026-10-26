# KDE number density, point masses and narrow kernels

With `normalize=False`, KDE output retains the sum of positive number-concentration
weights; a restricted query interval retains only the density supported there.
With `normalize=True`, the distribution helper normalizes a positive integral.
Normalization requires at least two strictly increasing query nodes with finite
spacing in the integration coordinate. A regular, unnormalized KDE may still be
evaluated at a single point or in arbitrary query order. Zero supported density
remains zero, including when all weights are zero or the evaluated tails underflow.
The INSA distribution additionally retains its existing peak-normalization rule;
an empty supported distribution stays zero.

## Point masses: single-valued populations and narrow kernels

A Gaussian kernel sampled at the query nodes integrates correctly only when it is
wide compared with the node spacing. Sampled with the trapezoidal rule at uniform
spacing `d`, a kernel of bandwidth `h` has a relative integration error of about
`2 exp(-2 pi^2 (h/d)^2)`: 3e-4 at `h = 2d/3`, 1.4% at `h = d/2` and 34% at
`h = 0.3d`. Below that, a node next to a sample sees the kernel's peak and a sample
between nodes is missed entirely.

Samples are therefore represented as point masses when all of them share one value,
or when the fitted bandwidth is below two thirds of the largest spacing of the
sorted query grid. Each sample's weight is split between its two neighbouring nodes
in proportion to proximity (linear binning) in the declared linear or logarithmic
integration coordinate. This keeps the trapezoidal integral equal to the supported
number concentration, including at endpoints, and keeps interior first moments. It
introduces no physical spread. Broader kernels are sampled directly, as before.

The diameter and INSA variables pass their outer bin edges as point-mass support.
A point between an outer edge and the nearest center is deposited at that center
with its count conserved under the same trapezoidal quadrature. This boundary
representation preserves count, but moves the first moment to the nearest node.
Points outside the support are dropped, with a `RuntimeWarning` giving the share of
the weight lost. Generic helper calls retain query-interval support unless
`point_mass_bounds` is supplied.

### Integrate with the trapezoidal rule, not a bin sum

The conservation above holds for `np.trapezoid(dens, u)` over the query
coordinates, which gives each end node half a cell. A bin-style sum, `sum(dens * du)`
with the full spacing at every node, counts an end node twice. For 1e9 m^-3 deposited
at the first node of a log grid:

| Integration | Result |
| --- | --- |
| `np.trapezoid(dens, np.log(D))` | 1.0e9 m^-3 |
| `np.sum(dens) * dlnD` | 2.0e9 m^-3 |

Histogram distributions use their bin widths instead; the two quadratures are
distinct.

## Zero-area INSA particles

Zero-area particles are excluded from the INSA log distribution, since ln(0) is
undefined. Absolute densities retain only the concentration of positive-area
particles; normalization applies to that positive-area subset. An all-inactive
population returns zeros for both normalized and absolute KDE output.

## Normalized distributions at extreme concentrations

Normalized KDEs use SciPy's unscaled probability density, and normalized point
masses start with unit weight. They are then normalized on the query grid as
before. Multiplying by total concentration and subsequently dividing it out
can underflow or overflow unnecessarily; raw number-density output still uses
the total concentration.

## Numerically dominant sample weights

Even distinct samples can have numerically singular weighted covariance when
one normalized weight is essentially one. If `1 - sum(p**2)` is no larger
than eight float64 epsilons, samples take the same conservative point-mass
path. Weights are rescaled before this check to avoid overflow/underflow in
the covariance decision. This is a numerical fallback, not removal of small
positive concentrations; deposition still uses their original weights (or
unit-total weights for normalized output). Other fitting ValueErrors propagate.
