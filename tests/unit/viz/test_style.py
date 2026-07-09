"""
Additional tests for part2pop.viz.style using the existing API.
"""
import pytest

from part2pop.viz import style


def test_geomdefaults_combos_lengths():
    gd = style.GeomDefaults(
        palette=["a", "b"],
        linestyles=["-", "--"],
        markers=["o", "s"],
    )

    combos_ls = gd.combos(use_linestyle=True, use_marker=False)
    combos_mk = gd.combos(use_linestyle=False, use_marker=True)
    combos_both = gd.combos(use_linestyle=True, use_marker=True)
    combos_none = gd.combos(use_linestyle=False, use_marker=False)

    assert len(combos_ls) == 4  # 2 colors * 2 linestyles
    assert len(combos_mk) == 4  # 2 colors * 2 markers
    assert len(combos_both) == 8  # 2 colors * 2 linestyles * 2 markers
    assert len(combos_none) == 2  # just colors


def test_style_manager_plan_filters_unknown_kwargs():
    sm = style.StyleManager(style.Theme(), deterministic=True)
    overrides = {"series": {"color": "#123456", "bogus": 1}}

    planned = sm.plan("line", ["series"], overrides=overrides)
    line_style = planned["series"]

    assert "color" in line_style
    assert "bogus" not in line_style


def test_plan_same_key_is_deterministic():
    sm = style.StyleManager(style.Theme())

    keys = ["series0"]
    style1 = sm.plan("scatter", keys)
    style2 = sm.plan("scatter", keys)

    assert style1 == style2


def test_plan_different_keys_get_different_styles():
    sm = style.StyleManager(style.Theme())

    styles = sm.plan("scatter", ["x0", "x1", "x2"])

    assert set(styles.keys()) == {"x0", "x1", "x2"}
    assert not (styles["x0"] == styles["x1"] == styles["x2"])


def test_plan_respects_overrides():
    sm = style.StyleManager(style.Theme())
    keys = ["my_series"]

    base = sm.plan("line", keys)["my_series"]
    overrides = {"my_series": {"color": "#123456", "marker": "x"}}
    styled = sm.plan("line", keys, overrides=overrides)["my_series"]

    assert styled["color"] == "#123456"
    assert styled["marker"] == "x"
    for key, value in base.items():
        if key in {"color", "marker"}:
            continue
        assert styled[key] == value


def test_deterministic_vs_order_sensitive_when_requested():
    keys = ["a", "b"]

    sm_det = style.StyleManager(style.Theme(), deterministic=True)
    style_a_first = sm_det.plan("scatter", keys)["a"]
    style_a_second = sm_det.plan("scatter", list(reversed(keys)))["a"]
    assert style_a_first == style_a_second

    sm_nondet = style.StyleManager(style.Theme(), deterministic=False)
    style_a_first_nd = sm_nondet.plan("scatter", keys)["a"]
    style_a_second_nd = sm_nondet.plan("scatter", list(reversed(keys)))["a"]
    assert style_a_first_nd != style_a_second_nd


def test_unknown_geom_raises_valueerror():
    sm = style.StyleManager(style.Theme())

    with pytest.raises(ValueError):
        sm.plan("totally_unknown_geom", ["foo"])


def test_cycle_linestyle_flag_produces_multiple_styles():
    sm = style.StyleManager(style.Theme(), deterministic=True)
    planned = sm.plan("line", ["a", "b", "c", "d"], cycle_linestyle=True)
    unique = {value["linestyle"] for value in planned.values()}

    assert len(unique) > 1


def test_line_defaults_keep_solid_when_not_cycling():
    sm = style.StyleManager(style.Theme(), deterministic=True)
    planned = sm.plan("line", ["only"])

    assert planned["only"]["linestyle"] == "-"
