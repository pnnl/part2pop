from part2pop.population.builder import build_population


def make_monodisperse_population(D_values=(100e-9, 150e-9), N_values=None):
    D_values = tuple(D_values)
    if N_values is None:
        N_values = [1e6 for _ in D_values]
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"] for _ in D_values],
        "aero_spec_fracs": [[0.3, 0.5, 0.2] for _ in D_values],
        "N": N_values,
        "D": list(D_values),
    }
    return build_population(cfg)