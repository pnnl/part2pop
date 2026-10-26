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

Humidity equilibration retains its existing water-mass side effects.
These entry points do not make all diagnostic calculations pure or change temperature conversion,
freezing geometry, rate formulas, stochastic treatment, or time stepping.

Check INSA distribution override forwarding separately for the same policy.
Configuration routing alone does not supply a nucleating_sites numerical
method; verify its availability and accepted scientific definition before use.

## Temperature and humidity integration

The analysis implementations also retain the temperature validation and unit
contracts described here. Zero Celsius is valid; temperatures are
converted once, must be finite and above absolute zero, and are validated
before water equilibration. Frozen/unfrozen fractions retain RH=None as a
request to use current water and RH=0 as an explicit dry equilibrium. P_frz and
avg_Jhet validate required RH through the existing shared validator. The nucleating_sites observable retains its separately documented provisional
scientific definition; configuration forwarding does not validate that model.
