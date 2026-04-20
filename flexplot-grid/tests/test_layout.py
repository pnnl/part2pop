from flexplot_grid.layout import make_grid

def test_make_grid_rows():
    # Example test for make_grid_rows
    scenarios = ['Scenario 1', 'Scenario 2']
    times = [0, 1, 2]
    mode = 'example_mode'
    variables = ['Variable 1', 'Variable 2']
    scenario_configs = {}
    plotting_function = lambda x: x  # Placeholder for actual plotting function
    variable_configs = {}
    figsize = (10, 5)

    grid = make_grid_rows(scenarios, times, mode, variables, scenario_configs, plotting_function, variable_configs, figsize)
    
    assert grid is not None
    assert len(grid) == len(scenarios)
    assert all(len(row) == len(times) for row in grid)

def test_make_grid_columns():
    # Example test for make_grid_columns
    variable_names = ['Variable 1', 'Variable 2']
    scenarios = ['Scenario 1', 'Scenario 2']
    mode = 'example_mode'
    fixed_time = 1
    plotting_function = lambda x: x  # Placeholder for actual plotting function
    variable_configs = {}
    figsize = (10, 5)

    grid = make_grid_columns(variable_names, scenarios, mode, fixed_time, plotting_function, variable_configs, figsize)
    
    assert grid is not None
    assert len(grid) == len(variable_names)
    assert all(len(col) == len(scenarios) for col in grid)