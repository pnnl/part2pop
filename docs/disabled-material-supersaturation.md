# Disabled materials and ice supersaturation

The existing get_aw rule treats a nonwater species as dissolved solute only
when both effective heterogeneous coefficients are NaN. Disabling just one
coefficient removes its heterogeneous rate without adding it to the solute
sum; disabling both can lower water activity. This is retained behavior, not
an independent physical solubility model or an inference about real OC.

PR #123's delta_aw <= 0 cutoff also applies to a remaining ice-active BC
component in such a solution. A concentrated NaN-disabled OC / active-BC
mixture can have zero heterogeneous rate even though BC still has positive
area and finite coefficients. The numeric regression records this exact rule.
PR #136's positive-rate area control instead uses a dilute, demonstrably ice-
supersaturated droplet. Do not remove the cutoff to make an area test pass.

Whether solubility and ice activity should use independent metadata remains a
separate scientific decision. No solubility rule is changed by this regression.
