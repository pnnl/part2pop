# Sampled lognormal sample counts

N_parts is a finite positive real whole-number count, supplied as a total scalar
or one count per mode. Integral numeric strings/floats remain accepted;
complex scalars (including zero-imaginary complex dtypes) and
nonintegral values such as 2.9 are rejected before generic config conversion,
instead of silently becoming 2. Zero, negative, nonfinite, nested and counts
outside the platform integer range are rejected. The caller's config is copied.

The bounded largest-remainder allocation, one sample per mode (including
zero-number modes), exact total sample count, and per-mode concentration
weights are unchanged for valid inputs. A scalar total must be at least the
number of configured modes. Whether to remove zero-number modes remains a
separate API decision. No new sample-size sufficiency or Monte Carlo accuracy
claim is made by integer validation.
