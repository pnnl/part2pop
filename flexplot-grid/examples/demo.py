from flexplot_grid import make_grid_rows, make_grid_columns
import matplotlib.pyplot as plt
import numpy as np

# Example data
scenarios = ['Scenario 1', 'Scenario 2']
times = ['Time 1', 'Time 2', 'Time 3']
variables = ['Variable A', 'Variable B']
data = {
    'Scenario 1': {
        'Time 1': [1, 2],
        'Time 2': [2, 3],
        'Time 3': [3, 4]
    },
    'Scenario 2': {
        'Time 1': [2, 3],
        'Time 2': [3, 4],
        'Time 3': [4, 5]
    }
}

# Function to plot data
def plot_data(ax, x, y, label):
    ax.plot(x, y, label=label)
    ax.set_title(label)

# Create grid layout using make_grid_rows
fig1, axes1 = make_grid_rows(scenarios, times, mode='time', variables=variables, plot_func=plot_data, figsize=(10, 6))

# Create grid layout using make_grid_columns
fig2, axes2 = make_grid_columns(variables, scenarios, mode='scenario', plot_func=plot_data, figsize=(10, 6))

plt.show()