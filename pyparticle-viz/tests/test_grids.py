import pytest
from PyParticle.viz.grids import create_scenario_grid, create_variable_grid

def test_create_scenario_grid():
    # Test for creating a grid where each row is a different scenario
    scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    variables = ['Variable A', 'Variable B']
    times = [1, 12, 24]
    
    grid = create_scenario_grid(scenarios, variables, times)
    
    assert len(grid) == len(scenarios)
    assert all(len(row) == len(times) for row in grid)
    assert grid[0][0] == (scenarios[0], variables[0], times[0])  # Example check

def test_create_variable_grid():
    # Test for creating a grid where each column is a different variable
    scenarios = ['Scenario 1']
    variables = ['Variable A', 'Variable B', 'Variable C']
    times = [1, 12, 24]
    
    grid = create_variable_grid(scenarios, variables, times)
    
    assert len(grid) == len(variables)
    assert all(len(row) == len(times) for row in grid)
    assert grid[0][0] == (scenarios[0], variables[0], times[0])  # Example check