Data prep migration
====================

Summary
-------
This change moves plot data assembly logic from `src/PyParticle/analysis.py` into
`src/PyParticle/viz/data_prep.py` and makes `analysis.py` thin wrappers that delegate
to the new module. The public plotting contract (compute_variable(..., return_plotdat=True))
is preserved.

Files changed
-------------
- Added: `src/PyParticle/viz/data_prep.py`
- Modified: `src/PyParticle/analysis.py` (wrappers)
- Added tests: `tests/unit/test_data_prep.py`

How to run tests
----------------
Activate your dev environment and run:

```bash
pytest -q tests/unit/test_data_prep.py
```

How to validate demos
---------------------
After activation, run the viz demo scripts under `examples/` (PartMC demo requires local PartMC outputs):

```bash
python examples/viz_grid_scenarios_variables.py
python examples/viz_grid_partmc_scenarios.py
```

Notes
-----
- Optical builder calls were left delegated to the existing optics builder via `build_optical_population`.
- The goal was minimal, safe refactor: behaviour should be unchanged; only the code location changed.
