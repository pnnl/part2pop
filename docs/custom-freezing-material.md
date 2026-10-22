# Active material, lookup failures and retained solubility policy

Positive finite material volume contributes to one pooled equivalent sphere
only with a complete finite heterogeneous coefficient pair. Zero/zero is valid;
NaN or infinity in either coefficient disables activity. Volume is selected by
species position. The rate mask is supplied separately by PR #123.

PR #135 supplies exact freezing-table name lookup and should merge first.
Missing names can use complete custom pairs; active-area lookup catches
ValueError (or the older UnboundLocalError for an unknown name), not arbitrary
I/O errors. An unreadable table therefore raises rather than silently removing
active material. The legacy reader also uses ValueError for malformed rows;
distinguishing malformed data from missing species needs a future reader API.

The shared active-area source in PRs #127 and #136 must be reconciled against
their current branch heads before merging the second PR. These follow-up
patches align their shared source, and compatibility validation covers both
orders; an earlier claim about unmodified heads merging cleanly was incorrect.

## Scientific decision still required: solubility versus activity

The water-activity calculation remains unchanged. It treats a nonwater species
as a solute only when BOTH effective heterogeneous coefficients are NaN.
Disabling just one coefficient removes activity without classifying it as a
solute. Disabling both also adds its molality to the osmotic sum, potentially
changing homogeneous freezing substantially. Neither PR keeps every disabled
material out of the solute calculation. Infinity and a partial NaN pair are
also distinct from a NaN/NaN pair under this legacy classification.

This documents existing behavior, not physical validation. Independent
solubility/activity metadata requires a separate reviewed scientific change.

PR #127 additionally forwards INSA_distribution species overrides: omitted or
None inherits population defaults; an explicit mapping replaces them, and {}
clears inheritance. The analysis does not persist the mapping on the population.
