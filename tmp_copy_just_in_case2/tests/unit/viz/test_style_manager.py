
# /tests/unit/viz/test_style_manager.py
import pytest

from part2pop.viz.style import StyleManager, Theme


def test_plan_same_key_is_deterministic():
    sm = StyleManager(Theme())

    keys = ["series0"]
    style1 = sm.plan("scatter", keys)
    style2 = sm.plan("scatter", keys)

    assert style1 == style2


def test_plan_different_keys_get_different_styles():
    sm = StyleManager(Theme())

    styles = sm.plan("scatter", ["x0", "x1", "x2"])
    assert set(styles.keys()) == {"x0", "x1", "x2"}
    # At least one style should differ across the three entries
    assert not (styles["x0"] == styles["x1"] == styles["x2"])


def test_plan_respects_overrides():
    sm = StyleManager(Theme())
    keys = ["my_series"]

    base = sm.plan("line", keys)["my_series"]
    overrides = {"my_series": {"color": "#123456", "marker": "x"}}
    styled = sm.plan("line", keys, overrides=overrides)["my_series"]

    assert styled["color"] == "#123456"
    assert styled["marker"] == "x"
    for k, v in base.items():
        if k in {"color", "marker"}:
            continue
        assert styled[k] == v


def test_deterministic_vs_order_sensitive_when_requested():
    keys = ["a", "b"]

    sm_det = StyleManager(Theme(), deterministic=True)
    style_a_first = sm_det.plan("scatter", keys)["a"]
    style_a_second = sm_det.plan("scatter", list(reversed(keys)))["a"]
    assert style_a_first == style_a_second

    sm_nondet = StyleManager(Theme(), deterministic=False)
    style_a_first_nd = sm_nondet.plan("scatter", keys)["a"]
    style_a_second_nd = sm_nondet.plan("scatter", list(reversed(keys)))["a"]
    assert style_a_first_nd != style_a_second_nd


def test_unknown_geom_raises_valueerror():
    sm = StyleManager(Theme())
    with pytest.raises(ValueError):
        sm.plan("totally_unknown_geom", ["foo"])


def test_cycle_linestyle_flag_produces_multiple_styles():
    sm = StyleManager(Theme(), deterministic=True)
    planned = sm.plan("line", ["a", "b", "c", "d"], cycle_linestyle=True)
    unique = {v["linestyle"] for v in planned.values()}

    assert len(unique) > 1


def test_line_defaults_keep_solid_when_not_cycling():
    sm = StyleManager(Theme(), deterministic=True)
    planned = sm.plan("line", ["only"])

    assert planned["only"]["linestyle"] == "-"
