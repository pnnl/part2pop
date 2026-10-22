# Rayleigh fallback acceptance policy

The proposed opt-in fallback accepts only homogeneous passive positive-index
mixtures with 1 <= Re(m) <= 2, 0 <= Im(m) <= 1, and external size parameter
x = 2*pi*r/lambda < 0.1 at every evaluated RH and wavelength. Refractive index,
wavelength and radius must be finite; wavelength must be positive and radius
nonnegative. Values outside this domain raise ValueError and require PyMieScatt.
The installed Mie path is unchanged and does not use these fallback-only limits.
Positive refractive-index bounds allow eight float64 epsilons of relative
roundoff, so volume averaging valid boundary components does not spuriously
reject their mixture. The zero lower bound on Im(m) remains strict; negative
absorption is not admitted by this allowance. Differences of 1e-10 outside the
positive bounds are still rejected. No wavelength or size-domain bound changes.

This is a conservative software acceptance policy for ordinary aerosol-like
indices, proposed for maintainer review. It is not a universal Rayleigh theorem
or a proof of an error bound throughout the continuum of accepted inputs.
At the largest accepted index magnitude, |m|*x < sqrt(5)*0.1, so the policy also
limits internal optical size. High-contrast, negative-index, gain-medium and
resonant cases are outside this fallback's supported domain.

Numerical characterization with PyMieScatt 1.8.1.1 sampled 21 real indices from
1 to 2, k=0 plus 16 logarithmic positive imaginary values from 1e-6 to 1,
and x in {0.001, 0.03, 0.06, 0.099999}: 1428 comparisons. For nonzero physical
signals, the largest relative differences were about 0.572% for scattering,
1.296% for absorption and 1.295% for extinction. Exactly m=1 has no particle/
medium contrast, and k=0 has no absorption; solver cancellation noise in those
zero signals is not a meaningful relative error. Committed boundary/interior
comparisons use a 2% relative tolerance for nonzero channels, with zero-channel
checks at a floating-point noise scale. These are empirical checks, not a
universal accuracy guarantee. No error guarantee is asserted for g=0.

Previously x alone admitted m=10+0.1i and m=30+0.1i at x=0.09. Their Rayleigh/
Mie absorption ratios were about 0.4734 and 0.000994. The constructor now rejects
both. A different supported domain requires an explicit reviewed policy and
new numerical validation; there is no bypass flag for unsupported indices.
The direct formula helper remains available for characterization experiments.

PR #105 contains the same fallback as #104; preserve this identical validation
when combining their source changes, along with #105's separate SSA cleanup.
