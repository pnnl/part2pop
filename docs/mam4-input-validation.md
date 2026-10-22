# MAM4 input validation

Number concentrations must be finite and nonnegative for every original mode,
before zero-number modes are filtered. A negative, NaN, infinite or masked
number is invalid data, not an inactive mode. Zero-number modes may omit their
composition; their diameter and composition fields are not reconstructed.

For active output modes, pressure (Pa), temperature (K) and converted dry-air
density must be finite positive real scalars. Complex scalar inputs are rejected
before float conversion can discard their imaginary parts. Dry diameters must be finite positive
and GSD must exceed one, matching the continuous binned-lognormal factory.
Component masses must be finite nonnegative and their reconstructed total
finite positive before normalization. Missing or masked required values fail.
Wet reconstruction additionally requires finite positive wet diameters >= dry
diameters and finite nonnegative water mass. No negative-value clipping,
NaN-to-zero replacement or measurement-noise tolerance is introduced. Dry-only
construction does not use or validate wet-diameter values.

Initial active-mode fractions use the same finite/nonnegative and positive-total
checks before normalization. Invalid inputs now raise ValueError rather than
silently dropping a mode or returning negative water. Valid inputs retain the
same normalization, air-density conversion and same-GSD moment assumptions.

This does not fix issue #137's diameter/mass/number timing mismatch or choose an
effective-density model. The timestep=1 namelist / timestep>=2 record n-2
mapping is retained. Scientific timing and example-configuration corrections
must be reviewed separately; simply shifting the timestep index is not a fix.
