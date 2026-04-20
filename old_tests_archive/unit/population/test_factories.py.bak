from __future__ import annotations
import pathlib, yaml, numpy as np, pytest

CASES_DIR = pathlib.Path(__file__).with_suffix("").parent / "cases"

def _iter_cases():
    return sorted(CASES_DIR.glob("*.yaml"))

def _load_case(path: pathlib.Path):
    cfg = yaml.safe_load(path.read_text()) or {}
    pop_cfg = cfg["population"]         # top-level 'type' inside (per repo)
    checks  = cfg.get("checks", {}) or {}
    # resolve any relative paths in config against the case file directory
    def _resolve(d):
        for k, v in list(d.items()):
            if isinstance(v, str) and ("/" in v or v.endswith((".csv",".nc",".json"))):
                d[k] = str((path.parent / v).resolve())
            elif isinstance(v, dict): _resolve(v)
            elif isinstance(v, list):
                for i, vv in enumerate(v):
                    if isinstance(vv, dict): _resolve(vv)
        return d
    return {"population": _resolve(pop_cfg), "checks": checks, "case_path": path}

def _assert_monotonic(arr, name):
    assert np.all(np.diff(arr) > 0), f"{name}: must be strictly increasing"

@pytest.mark.parametrize("case_file", _iter_cases(), ids=lambda p: p.name)
def test_population_case(case_file):
    from PyParticle.population.builder import build_population
    case = _load_case(case_file)
    pop = build_population(case["population"])

    assert hasattr(pop, "ids") and len(pop.ids) > 0
    Ds = np.array([pop.get_particle(pid).get_Ddry() for pid in pop.ids])

    ch = case["checks"]
    if ch.get("no_nans", True):       assert np.isfinite(Ds).all(), "Ddry: NaN/Inf"
    if ch.get("nonnegative", True):   assert (Ds > 0).all(), "Ddry: non-positive"
    if ch.get("units_meters", True):  assert (Ds > 1e-9).all() and (Ds < 1e-4).all(), "Ddry not in meters"
    if ch.get("diameters_monotonic", True): _assert_monotonic(Ds, "Ddry_m")
