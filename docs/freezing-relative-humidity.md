# Relative humidity in freezing analyses

RH is a scalar finite fraction in [0, 1], not a percentage. P_frz and avg_Jhet
always equilibrate water and reject None. Frozen/unfrozen fraction analyses
can use RH=None to retain existing water content; PR #116 validates explicit
RH, including zero. Invalid RH must be rejected before modifying water.

The retained Particle._equilibrate_h2o implementation caps RH above 0.99.
Therefore RH=1 requests water at RH=0.99; it does not calculate a saturated
equilibrium. This follow-up preserves that ceiling and introduces no new
clipping, warning or saturation treatment for valid fractional RH.

Maintain a separate scientific/API decision for rejecting, warning about or
modeling the interval above 0.99. avg_Jhet equilibrates once at its first grid
temperature and retains that water content while evaluating the temperature
grid. These analyses retain existing in-place water changes on valid calls.
