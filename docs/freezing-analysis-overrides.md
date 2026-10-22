# Freezing analysis species overrides

`P_frz`, `avg_Jhet`, `frozen_frac`, `unfrozen_frac`, and `nucleating_sites`
use the same authoritative species-modification policy:

| Analysis configuration | Effective mapping |
| --- | --- |
| Key omitted | Population `species_modifications` |
| `species_modifications=None` | Population `species_modifications` |
| `species_modifications={}` | Empty mapping; tabulated/default coefficients |
| Explicit nonempty mapping | Exactly that mapping; no per-species merge |

Overrides apply to one calculation. These analysis entry points do not replace
the population's modification mapping or edit either input mapping. The shared
resolver creates an analysis-local configuration and borrows the selected mapping;
freezing construction treats that mapping as read-only. Direct freezing-builder
APIs retain their existing behavior.

Humidity equilibration retains its existing water-mass side effects. This patch
does not make all diagnostic calculations pure or change temperature conversion,
freezing geometry, rate formulas, stochastic treatment, or time stepping.

The separate P05 custom-area patch gives `INSA_distribution` this same policy;
apply both to cover all six analysis entry points. P11 is independently applicable
to the pinned upstream base and supplies the other five entry points. The missing
nucleating-site numerical API is repaired by existing PR #117, not by this patch.

## Temperature and humidity integration

The analysis implementations also retain the temperature validation and unit
contracts of PRs #111, #116 and #131. Zero Celsius is valid; temperatures are
converted once, must be finite and above absolute zero, and are validated
before water equilibration. Frozen/unfrozen fractions retain RH=None as a
request to use current water and RH=0 as an explicit dry equilibrium. P_frz and
avg_Jhet validate required RH through the existing shared validator. Shared
regression files are byte-identical across the overlapping PRs and should be
kept once. The nucleating_sites numerical method still comes from PR #117;
this patch does not choose or duplicate its provisional scientific definition.
