# MAM4 coagulation-only regression data

The two directories below contain raw `mam_output.nc` files and the matching
MAM4 `namelist` from the AMBRS tutorial's coagulation-only scenarios:

- `coagulation-only-base-case` (`numc1 = 5e12`)
- `coagulation-only-high-particle-count` (`numc1 = 5e14`)

They were produced with the AMBRS tutorial configuration using coagulation
only (`mdo_coag=1`, with gas chemistry, gas/aerosol exchange, and nucleation
disabled), and an initial dry composition of 45% SO4 / 55% POM by mass.
The files are intentionally raw model output; `part2pop` does not modify the
MAM4 box model or AMBRS.  POM is absent from the MAM4 output, so the factory
retains that contribution only for runs whose namelist proves the
coagulation-only conditions and a common SO4/POM ratio.  This inference is
not valid for condensation, chemistry, nucleation, or changing composition.
