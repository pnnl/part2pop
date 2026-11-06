# Compatibility and numerical changes

Effective radius is sum(N*r**3)/sum(N*r**2), using current wet radii in
meters, rather than the previous number-mean radius. Polydisperse results
change; monodisperse values agree. Water is not re-equilibrated. Zero total
area returns NaN. Recheck downstream comparisons against the historical
number-mean statistic and retain the spherical-particle assumptions.
