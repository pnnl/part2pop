from __future__ import annotations
from typing import Dict, Any, Iterable
import importlib, numpy as np, pytest
from ..checks import (
    assert_no_nans, assert_nonnegative, assert_monotonic_increasing, assert_in_unit_interval,
    load_expected_csv, write_diff_csv, DEFAULT_RTOL, DEFAULT_ATOL
)

def _load_adapter(path: str):
    mod_path, fn = path.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn)


def run_optics(cfg: Dict[str, Any], base_dir, tmp_dir):
    """
    Build population from cfg['population'] (top-level 'type'),
    build optical population from cfg['optics'] (must include 'type' morphology),
    then compute requested analysis variables (default: b_scat, b_abs, b_ext).
    """
    from PyParticle.population.builder import build_population
    from PyParticle.optics.builder import build_optical_population
    from PyParticle.analysis.population.factory import b_scat as _bscat
    from PyParticle.analysis.population.factory import b_abs as _babs
    from PyParticle.analysis.population.factory import b_ext as _bext

    pop_cfg   = cfg["population"]
    optics_cfg = cfg.get("optics") or {}
    if "type" not in optics_cfg:
        raise AssertionError("optics config must include 'type' (morphology), e.g. 'homogeneous'.")

    population = build_population(pop_cfg)

    # Variables: compute via the variable builders' compute(pop) API
    variables = (cfg.get("analysis") or {}).get("variables", ["b_scat","b_abs","b_ext"])
    # Build optics-aware cfg expected by the analysis variables' compute()
    var_base_cfg = {
        "morphology": optics_cfg.get("type", "homogeneous"),
        "rh_grid": list(optics_cfg.get("rh_grid", [0.0])),
        "wvl_grid": list(optics_cfg.get("wvl_grid", optics_cfg.get("wvls", [550e-9]))),
        "T": float(optics_cfg.get("temp", optics_cfg.get("T", 298.15))),
        "species_modifications": optics_cfg.get("species_modifications", {}),
    }

    results = {}
    for v in variables:
        if v == "b_scat":
            var = _bscat.build(var_base_cfg.copy())
        elif v == "b_abs":
            var = _babs.build(var_base_cfg.copy())
        elif v == "b_ext":
            var = _bext.build(var_base_cfg.copy())
        else:
            raise AssertionError(f"Unknown analysis variable: {v}")
        # analysis variables expect a base population and build optics internally
        arr = var.compute(population)
        arr = np.asarray(arr, dtype=float)
        assert_no_nans(v, arr); assert_nonnegative(v, arr)
        results[v] = arr

    # Grid checks
    wvl = np.asarray(optics_cfg.get("wvl_grid", [550e-9]), dtype=float)
    rh  = np.asarray(optics_cfg.get("rh_grid", [0.0]), dtype=float)
    assert_monotonic_increasing("wvl_grid (m)", wvl)
    assert_in_unit_interval("rh_grid", rh)

    # Extinction balance
    inv = (cfg.get("checks", {}).get("invariants", {}) or {})
    ext = inv.get("extinction_balance", {"enabled": True})
    if isinstance(ext, dict) and ext.get("enabled", True):
        if {"b_ext","b_scat","b_abs"}.issubset(results):
            lhs = results["b_ext"]
            rhs = results["b_scat"] + results["b_abs"]
            rtol = ext.get("rtol", DEFAULT_RTOL); atol = ext.get("atol", DEFAULT_ATOL)
            assert np.allclose(lhs, rhs, rtol=rtol, atol=atol), "b_ext != b_scat + b_abs"

    got = {k: np.asarray(v).squeeze() for k, v in results.items()}

    # Gold first
    expected_csv = cfg.get("expected_csv") or (cfg.get("checks", {}) or {}).get("expected_csv")
    if expected_csv:
        exp = load_expected_csv(str((base_dir / expected_csv).resolve()))
        diff = write_diff_csv(tmp_dir, "optics_gold", got, exp)
        for k in set(got) & set(exp):
            assert np.allclose(got[k], exp[k], rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL), \
                f"{cfg.get('name','<scenario>')}: gold mismatch for {k}. See {diff}"
        return

    # Side-by-side reference (optional)
    ref = cfg.get("reference") or {}
    if "adapter" in ref:
        try:
            run_ref = _load_adapter(ref["adapter"])
        except Exception as e:
            pytest.skip(f"Reference adapter unavailable: {e}")
        ref_out = run_ref(cfg)  # adapter sees whole scenario
        ref_out = {k: np.asarray(v).squeeze() for k, v in ref_out.items()}
        rtol = ref.get("rtol", 0.08); atol = ref.get("atol", 0.0)
        diff = write_diff_csv(tmp_dir, "optics_ref", got, ref_out)
        for k in set(got) & set(ref_out):
            assert np.allclose(got[k], ref_out[k], rtol=rtol, atol=atol), \
                f"{cfg.get('name','<scenario>')}: reference mismatch for {k}. See {diff}"
