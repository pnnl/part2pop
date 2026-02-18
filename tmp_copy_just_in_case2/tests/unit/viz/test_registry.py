from part2pop.viz.factory.registry import discover_plotter_types


def test_discover_plotter_types_finds_registered_modules():
    types = discover_plotter_types()

    # Built-in plotter modules should be discovered
    assert "state_line" in types
    assert "state_scatter" in types
    assert callable(types["state_line"])
    assert callable(types["state_scatter"])
