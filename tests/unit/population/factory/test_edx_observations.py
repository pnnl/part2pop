import numpy as np

from part2pop import ParticlePopulation, build_population


def _write_edx_csv(tmp_path):
    csv_path = tmp_path / "edx_small.csv"
    # Keep columns aligned with current reader expectations:
    # - element columns by direct names
    # - one diameter column containing "diam"
    # - one class/type column containing "class"
    csv_path.write_text(
        "diam_um,class,C,N,O,Na,Mg,Al,Si,P,S,Cl,K,Ca,Mn,Fe,Zn\n"
        # biological row tuned to current mapping logic:
        # - sums to 1.0
        # - keeps Al/Si near zero so OIN contribution stays small
        # - leaves substantial C/N/O/P/S pool for biological fraction
        "0.5,biological,0.35,0.12,0.33,0.04,0.04,0.00,0.00,0.04,0.04,0.04,0.00,0.00,0.00,0.00,0.00\n"
        "",
        encoding="utf-8",
    )
    return csv_path


def test_edx_observations_smoke_build_population(tmp_path):
    edx_csv = _write_edx_csv(tmp_path)
    cfg = {
        "type": "edx_observations",
        "edx_file": str(edx_csv),
    }

    pop = build_population(cfg)
    assert isinstance(pop, ParticlePopulation)

    # Characterize current behavior only
    assert hasattr(pop, "classes")
    assert len(pop.classes) == len(pop.ids)
    assert len(pop.ids) == 1
    assert len(pop.classes) == 1

    # Number concentrations and diameters should be positive/plausible
    assert np.all(pop.num_concs > 0)
    d = pop.get_particle_var("Ddry")
    assert np.all(np.isfinite(d))
    assert np.all(d > 0)
