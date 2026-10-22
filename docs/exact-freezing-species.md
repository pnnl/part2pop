# Exact freezing-table species lookup

The freezing-data reader matches the full first token in each row,
case-insensitively. A custom species named `B` must not inherit coefficients
from `BC` or `biological`, and `O` must not inherit coefficients from `OC`.
Blank lines, comments and indentation are ignored when finding the row.

A missing exact name raises `ValueError`. The existing homogeneous-particle
fallback then uses explicitly supplied custom coefficients, or keeps the
material inactive when no complete pair is available. A partial custom pair
does not borrow its missing coefficient from a prefix-matched table species.
Exact tabulated names retain their existing coefficients and override policy,
including zero and NaN overrides; aliases remain the species registry's job.

This lookup patch is independent on main. P05 supplies the separate custom-area
behavior and P12 supplies disabled-material area consistency. Apply the default
series for all three behaviors together. This does not change the retained
coefficient-based water-activity classification or the table's scientific fits.
