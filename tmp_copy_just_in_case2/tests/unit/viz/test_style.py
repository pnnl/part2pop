"""
Additional tests for part2pop.viz.style using the existing API.
"""
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
