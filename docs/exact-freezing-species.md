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

Exact table-name lookup, particle-property name lookup and active-area
selection are separate contracts. Exercise them together for custom names,
complete and partial coefficient pairs, and disabled materials. Exact matching
does not change coefficient-based water-activity classification or the table's
scientific fits.
