# Population species alignment

Adding or replacing a particle preserves the established species-column order.
Compatible permutations are reordered by exact, case-sensitive species name;
missing, extra or duplicate names and mismatched mass-row shapes are rejected
before mutation. Available density, kappa, molar mass and surface tension must
match exactly; two NaN (unset) values count as equal. Runtime caches are not species identity; name-only lightweight
objects remain supported. This does not reconcile different physical definitions
or expand the population species set.

Tuple and list IDs are accepted. Replacing a particle retains its existing ID;
appending builds a new ID list. Replacement promotes integer-backed mass and
concentration arrays to floating point to preserve fractional values. Existing
extended-precision floating arrays and incoming masses retain their precision.
Replacement chooses a common dtype from both the existing arrays and incoming
mass/concentration values, including when the existing arrays are float64.
These methods
assume an otherwise consistent population and scalar numeric concentration;
it does not add general validation of IDs, classes, masses or physical fields.

## Test/documentation gap: public constructor precision

The precision guarantee starts with the values already present in an incoming
object's `masses` array. The public `Particle` constructor currently converts
masses to float64 before these population methods see them, and the population
methods cannot restore precision lost in that earlier conversion.

The extended-precision tests intentionally use lightweight particle-like
objects to exercise population storage independently. They do not establish an
end-to-end long-double guarantee through the public constructor. Such a
guarantee requires a separate constructor policy decision and tests using real
`Particle` instances; the constructor is unchanged.
