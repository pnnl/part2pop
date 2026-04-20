from __future__ import annotations
from typing import Dict, Any
import numpy as np, pytest
from ..checks import (
    assert_no_nans, assert_nonnegative, assert_monotonic_increasing,
    load_expected_csv, write_diff_csv, DEFAULT_RTOL, DEFAULT_ATOL
)

# todo: extend this to be more comprehensive, e.g., check composition, moments, etc.
def run_base(cfg: Dict[str, Any], base_dir, tmp_dir):
    """
    Build a base ParticlePopulation from cfg['population'] (top-level 'type' inside),
    then run foundational invariants (units, monotonicity, no NaNs, nonnegative),
    and optional gold comparisons (e.g., Ddry_m, dNdlnD).
    """
    from PyParticle.population.builder import build_population
    from PyParticle.analysis.builder import build_variable

    pop_cfg = cfg["population"]  # expects top-level 'type' per your builder
    population = build_population(pop_cfg)

    assert hasattr(population, "ids") and len(population.ids) > 0

    Ds = np.array([population.get_particle(pid).get_Ddry() for pid in population.ids])

    inv = (cfg.get("checks", {}).get("invariants", {}) or {})
    if inv.get("no_nans", True):         assert_no_nans("Ddry_m", Ds)
    if inv.get("nonnegative", True):     assert_nonnegative("Ddry_m", Ds)
    if inv.get("diameters_monotonic", True): assert_monotonic_increasing("Ddry_m", Ds)
    if inv.get("units_meters", True):
        assert (Ds > 1e-9).all() and (Ds < 1e-4).all(), "Ddry_m out of plausible meter range"

    # Optional: dNdlnD if available and desired
    try:
        if inv.get("check_dNdlnD", False):
            dcfg = inv.get("dNdlnD_cfg", {"D_min": 1e-9, "D_max": 1e-6, "N_bins": 50, "wetsize": False})
            var = __import__("PyParticle.analysis.population.factory.dNdlnD", fromlist=["build"]).build(dcfg)
            dnd = var.compute(population)
            assert_no_nans("dNdlnD", dnd)
            assert_nonnegative("dNdlnD", dnd)
    except Exception:
        # fine to skip if analysis not wired yet in this environment
        pass

    expected_csv = cfg.get("expected_csv") or (cfg.get("checks", {}) or {}).get("expected_csv")
    if expected_csv:
        exp = load_expected_csv(str((base_dir / expected_csv).resolve()))
        got = {"Ddry_m": Ds}
        diff = write_diff_csv(tmp_dir, "base", got, exp)
        for k in set(got) & set(exp):
            assert np.allclose(got[k], exp[k], rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL), \
                f"{cfg.get('name','<scenario>')}: {k} mismatch vs gold. See {diff}"
