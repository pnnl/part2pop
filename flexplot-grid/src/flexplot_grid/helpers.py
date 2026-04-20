"""
This file is part of the FlexPlot Grid Package.

Role:
- helpers: Additional helper functions that support the grid layout helpers.
"""

from typing import Callable, List, Any
import matplotlib.pyplot as plt

def make_grid_rows(scenarios: List[str], times: List[str], mode: str, variables: List[str],
                   scenario_configs: List[dict], plot_func: Callable, var_configs: List[dict],
                   figsize: tuple = (10, 8)):
    """Creates a grid where each row represents a scenario or variable, with columns for time snapshots.

    Parameters:
    - scenarios: List of scenario names.
    - times: List of time snapshots.
    - mode: Mode of plotting (e.g., 'line', 'bar').
    - variables: List of variable names to plot.
    - scenario_configs: List of configurations for each scenario.
    - plot_func: Function to call for plotting.
    - var_configs: List of configurations for each variable.
    - figsize: Size of the figure.

    Returns:
    - fig: The created figure.
    - axes: The array of axes for the subplots.
    """
    fig, axes = plt.subplots(len(scenarios), len(times), figsize=figsize)
    for i, scenario in enumerate(scenarios):
        for j, time in enumerate(times):
            for variable, config in zip(variables, var_configs):
                plot_func(axes[i, j], scenario, time, variable, config)
    plt.tight_layout()
    return fig, axes

def make_grid_columns(variable_names: List[str], scenarios: List[str], mode: str, 
                      fixed_time: str, plot_func: Callable, var_configs: List[dict],
                      figsize: tuple = (10, 8)):
    """Creates a grid where each column represents a variable, with rows for scenarios or times.

    Parameters:
    - variable_names: List of variable names to plot.
    - scenarios: List of scenario names or times.
    - mode: Mode of plotting (e.g., 'line', 'bar').
    - fixed_time: The fixed time snapshot for the plots.
    - plot_func: Function to call for plotting.
    - var_configs: List of configurations for each variable.
    - figsize: Size of the figure.

    Returns:
    - fig: The created figure.
    - axes: The array of axes for the subplots.
    """
    fig, axes = plt.subplots(len(variable_names), len(scenarios), figsize=figsize)
    for i, variable in enumerate(variable_names):
        for j, scenario in enumerate(scenarios):
            plot_func(axes[i, j], scenario, fixed_time, variable, var_configs[i])
    plt.tight_layout()
    return fig, axes
"""