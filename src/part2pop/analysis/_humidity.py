"""Input validation for analyses that always equilibrate particle water."""
import numpy as np


def validate_equilibration_rh(value):
    """Require a scalar finite RH fraction, retaining the existing 0.99 cap.

    Values in [0, 1] are accepted; Particle._equilibrate_h2o still caps values
    above 0.99 internally. This validates units/range, not saturation physics.
    Optional-equilibration analyses handle None separately (PR #116).
    """
    if np.ndim(value) != 0:
        raise ValueError("RH must be a scalar fraction between 0 and 1")
    try:
        rh = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RH must be a finite fraction between 0 and 1") from exc
    if not np.isfinite(rh) or rh < 0 or rh > 1:
        raise ValueError(f"RH must be a finite fraction between 0 and 1, got {value!r}; give 85% as 0.85")
    return rh
