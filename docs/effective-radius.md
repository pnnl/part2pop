# Population effective radius

`ParticlePopulation.get_effective_radius()` uses wet radii in meters and returns
`sum(N*r**3) / sum(N*r**2)`, the surface-area-weighted effective radius.
This differs from the number-mean radius for polydisperse populations. With no
total particle area (including an empty or zero-concentration population), the
result is undefined and returned as NaN.

The radii come from each particle's current water content: the method does
not equilibrate water to any humidity, so call the population's humidity
equilibration first when a specific RH is intended.

**Behavior change:** earlier releases returned the number-mean radius,
`sum(N*r) / sum(N)`. Results for polydisperse populations change, and are
larger, because the area weighting favors larger particles.
