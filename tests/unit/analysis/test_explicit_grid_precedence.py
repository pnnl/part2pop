import numpy as np
import pytest
from part2pop.analysis.builder import VariableBuilder, build_variable
from part2pop.viz.factory.state_line import StateLinePlotter

from part2pop import ParticlePopulation, make_particle


def population():
    p = make_particle(100e-9, ["SO4"], [1.0])
    return ParticlePopulation(p.species, np.array([p.masses]), np.array([600.0]), [0])


@pytest.mark.parametrize("name", ["Nccn", "frac_ccn", "s_grid"])
def test_supersaturation_alias_overrides_default(name):
    v = build_variable(name, var_cfg={"s_eval": [0.2, 0.4]})
    if name == "s_grid":
        out = v.compute(population())
    else:
        out = v.compute(population(), as_dict=True)["s"]
    np.testing.assert_array_equal(out, [0.2, 0.4])


def test_wavelength_alias_reaches_real_optics():
    v = build_variable(
        "b_abs",
        var_cfg={
            "wvls": [400e-9, 700e-9],
            "rh_grid": [0.0],
            "morphology": "homogeneous",
        },
    )
    np.testing.assert_array_equal(
        v.compute(population(), as_dict=True)["wvl_grid"], [400e-9, 700e-9]
    )


def test_bins_and_real_distribution_plot_share_requested_grid():
    cfg = {"N_bins": 3, "D_min": 20e-9, "D_max": 500e-9}
    p = population()
    v = build_variable("dNdlnD", var_cfg=cfg)
    out = v.compute(p, as_dict=True)
    np.testing.assert_allclose(
        out["edges"], np.geomspace(20e-9, 500e-9, 4), rtol=1e-12, atol=0.0
    )
    prepared = StateLinePlotter({"varname": "dNdlnD", "var_cfg": cfg}).prep(p)
    np.testing.assert_array_equal(prepared["x"], out["D"])
    np.testing.assert_array_equal(prepared["y"], out["dNdlnD"])


def test_explicit_edges_control_plot_coordinates():
    cfg = {"edges": np.geomspace(20e-9, 500e-9, 4)}
    out = build_variable("dNdlnD", var_cfg=cfg).compute(population(), as_dict=True)
    prepared = StateLinePlotter({"varname": "dNdlnD", "var_cfg": cfg}).prep(
        population()
    )
    np.testing.assert_array_equal(prepared["x"], out["D"])


def test_modify_alias_wins_over_lower_explicit_canonical():
    v = (
        VariableBuilder("Nccn", {"s_grid": [0.1, 0.2]})
        .modify(s_eval=[0.3, 0.4])
        .build()
    )
    np.testing.assert_array_equal(
        v.compute(population(), as_dict=True)["s"], [0.3, 0.4]
    )


def test_existing_same_layer_precedence_and_defaults_remain():
    v = build_variable("Nccn", var_cfg={"s_grid": [0.1, 0.2], "s_eval": [0.3, 0.4]})
    np.testing.assert_array_equal(
        v.compute(population(), as_dict=True)["s"], [0.1, 0.2]
    )
    assert build_variable("dNdlnD").compute(population()).shape == (80,)
    assert build_variable("diam_grid", var_cfg={"N_bins": 3}).compute(
        population()
    ).shape == (80,)


def test_insa_bin_controls_and_plot_coordinates():
    p = make_particle(100e-9, ["OC"], [1.0])
    pop = ParticlePopulation(p.species, np.array([p.masses]), np.array([1.0]), [0])
    cfg = {"method": "hist", "N_bins": 4, "INSA_min": 1e-15, "INSA_max": 1e-12}
    out = build_variable("INSA_distribution", var_cfg=cfg).compute(pop, as_dict=True)
    assert out["INSA"].shape == (4,)
    prepared = StateLinePlotter({"varname": "INSA_distribution", "var_cfg": cfg}).prep(
        pop
    )
    np.testing.assert_array_equal(prepared["x"], out["INSA"])
    np.testing.assert_array_equal(prepared["y"], out["dNdlnINSA"])


def test_none_edges_do_not_hide_explicit_bin_controls():
    v = build_variable("dNdlnD", var_cfg={"edges": None, "N_bins": 3})
    assert v.compute(population()).shape == (3,)


def test_modify_canonical_grid_wins_over_lower_alias_without_mutating_input():
    cfg = {"s_eval": [0.1, 0.2]}
    v = VariableBuilder("Nccn", cfg).modify(s_grid=[0.3, 0.4]).build()
    np.testing.assert_array_equal(
        v.compute(population(), as_dict=True)["s"], [0.3, 0.4]
    )
    assert cfg == {"s_eval": [0.1, 0.2]}


@pytest.mark.parametrize(
    "direction", ["edges_to_centers", "centers_to_edges", "edges_to_bins"]
)
def test_runtime_grid_representation_overrides_lower_explicit_representation(direction):
    centers = np.geomspace(30e-9, 300e-9, 3)
    edges = np.geomspace(20e-9, 500e-9, 5)
    if direction == "edges_to_centers":
        v = (
            VariableBuilder("dNdlnD", {"edges": edges})
            .modify(diam_grid=centers)
            .build()
        )
        np.testing.assert_allclose(
            v.compute(population(), as_dict=True)["D"], centers, rtol=1e-12, atol=0.0
        )
    else:
        cfg = (
            {"diam_grid": centers}
            if direction == "centers_to_edges"
            else {"edges": edges}
        )
        mod = (
            {"edges": edges}
            if direction == "centers_to_edges"
            else {"N_bins": 2, "D_min": 20e-9, "D_max": 500e-9}
        )
        v = VariableBuilder("dNdlnD", cfg).modify(**mod).build()
        expected = (
            edges if direction == "centers_to_edges" else np.geomspace(20e-9, 500e-9, 3)
        )
        np.testing.assert_allclose(
            v.compute(population(), as_dict=True)["edges"],
            expected,
            rtol=1e-12,
            atol=0.0,
        )


def test_provided_distribution_plot_uses_returned_coordinates(monkeypatch):
    pop = population()
    provided = {
        "D": np.array([50e-9, 80e-9, 160e-9]),
        "dNdlnD": np.array([1.0, 3.0, 2.0]),
    }
    monkeypatch.setattr(pop, "get_provided_dNdlnD", lambda cfg: provided, raising=False)
    prepared = StateLinePlotter(
        {"varname": "dNdlnD", "var_cfg": {"method": "provided"}}
    ).prep(pop)
    np.testing.assert_array_equal(prepared["x"], provided["D"])
    np.testing.assert_array_equal(prepared["y"], provided["dNdlnD"])


@pytest.mark.parametrize(
    "highest", ["canonical_defaults", "local_defaults", "user", "runtime"]
)
def test_grid_precedence_includes_variable_local_defaults(monkeypatch, highest):
    from types import SimpleNamespace
    from part2pop.analysis import builder as builder_module

    global_defaults = {"s_grid": [0.1, 0.2]}
    local_defaults = {"s_eval": [0.3, 0.4]} if highest != "canonical_defaults" else {}
    user = {"s_grid": [0.5, 0.6]} if highest in ("user", "runtime") else {}
    runtime = {"s_eval": [0.7, 0.8]} if highest == "runtime" else {}

    class Variable:
        meta = SimpleNamespace(default_cfg=local_defaults)

        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setattr(
        builder_module, "_get_defaults_for_var", lambda name: global_defaults
    )
    monkeypatch.setattr(
        builder_module, "_get_registry_builder", lambda scope: lambda name: Variable
    )
    variable = VariableBuilder("Nccn", user).modify(**runtime).build()
    expected = {
        "canonical_defaults": [0.1, 0.2],
        "local_defaults": [0.3, 0.4],
        "user": [0.5, 0.6],
        "runtime": [0.7, 0.8],
    }[highest]
    np.testing.assert_array_equal(variable.cfg["s_grid"], expected)
    np.testing.assert_array_equal(variable.cfg["s_eval"], expected)
    assert global_defaults == {"s_grid": [0.1, 0.2]}


def test_missing_defaults_provider_still_honors_explicit_grid(monkeypatch):
    from part2pop.analysis import builder as builder_module

    def unavailable(name):
        raise RuntimeError("defaults unavailable")

    monkeypatch.setattr(builder_module, "_get_defaults_for_var", unavailable)
    variable = build_variable("Nccn", var_cfg={"s_eval": [0.2, 0.4]})
    np.testing.assert_array_equal(
        variable.compute(population(), as_dict=True)["s"], [0.2, 0.4]
    )


def test_none_defaults_provider_still_honors_explicit_grid(monkeypatch):
    from part2pop.analysis import builder as builder_module

    monkeypatch.setattr(builder_module, "_get_defaults_for_var", lambda name: None)
    variable = build_variable("Nccn", var_cfg={"s_eval": [0.2, 0.4]})
    np.testing.assert_array_equal(
        variable.compute(population(), as_dict=True)["s"], [0.2, 0.4]
    )


def test_irregular_histogram_grid_reports_centers_of_its_actual_bins():
    requested = np.array([1e-8, 2e-8, 1e-7, 1e-6])
    cfg = {"method": "hist", "diam_grid": requested}
    computed = build_variable("dNdlnD", var_cfg=cfg).compute(population(), as_dict=True)
    np.testing.assert_allclose(
        computed["D"], np.sqrt(computed["edges"][:-1] * computed["edges"][1:])
    )
    plotted = StateLinePlotter({"varname": "dNdlnD", "var_cfg": cfg}).prep(population())
    np.testing.assert_array_equal(plotted["x"], computed["D"])
    np.testing.assert_array_equal(plotted["y"], computed["dNdlnD"])


@pytest.mark.parametrize("representation", ["centers", "edges", "bins"])
def test_insa_runtime_grid_representation_is_authoritative(representation):
    centers = np.geomspace(1e-15, 1e-12, 3)
    edges = np.geomspace(1e-15, 1e-12, 5)
    if representation == "centers":
        variable = (
            VariableBuilder("INSA_distribution", {"method": "hist", "edges": edges})
            .modify(insa_grid=centers)
            .build()
        )
        out = variable.compute(population(), as_dict=True)
        np.testing.assert_allclose(out["INSA"], centers, rtol=1e-12, atol=0.0)
    else:
        controls = (
            {"edges": edges}
            if representation == "edges"
            else {"N_bins": 2, "INSA_min": 1e-15, "INSA_max": 1e-12}
        )
        variable = (
            VariableBuilder(
                "INSA_distribution", {"method": "hist", "insa_grid": centers}
            )
            .modify(**controls)
            .build()
        )
        out = variable.compute(population(), as_dict=True)
        expected = edges if representation == "edges" else np.geomspace(1e-15, 1e-12, 3)
        np.testing.assert_allclose(out["edges"], expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "name,center_key", [("dNdlnD", "diam_grid"), ("INSA_distribution", "insa_grid")]
)
def test_same_layer_explicit_centers_and_edges_are_preserved(name, center_key):
    edges = (
        np.geomspace(1e-15, 1e-12, 4)
        if center_key == "insa_grid"
        else np.geomspace(20e-9, 500e-9, 4)
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    variable = build_variable(
        name,
        var_cfg={"method": "hist", "edges": edges, center_key: centers, "N_bins": 99},
    )
    np.testing.assert_array_equal(variable.cfg[center_key], centers)
    out = variable.compute(population(), as_dict=True)
    np.testing.assert_array_equal(out["edges"], edges)


def test_same_layer_wavelength_canonical_precedence_is_preserved():
    variable = build_variable(
        "wvl_grid", var_cfg={"wvl_grid": [400e-9, 700e-9], "wvls": [500e-9]}
    )
    np.testing.assert_array_equal(variable.compute(population()), [400e-9, 700e-9])
