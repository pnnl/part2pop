# EDX input compatibility

Population_MassFracs remains an alias of EdxElementMassFractions, including the
positional constructor and fields. Prefer the descriptive name in new code;
the alias is retained for at least one release.

Elemental fractions must be finite and nonnegative. No values are silently
clipped or filled. Preprocess instrument noise explicitly, retain the raw
measurements and document any threshold and resulting mass-closure change.
Elemental measurements do not uniquely identify chemical compounds; the
existing sulfate/oxide/organic reconstruction remains an approximation.

The parity fixture changed its dust row from O=0.39, Cu=0 to O=0.38, Cu=0.01
to exercise Cu allocation while retaining a valid reconstructed mixture. The
original row remains covered separately: its finite input is accepted by the
parser and its reconstruction outcome is checked explicitly.
