# flexplot-grid - Flexible Plotting Package

This project provides a flexible plotting package that allows users to create structured grid layouts for visualizing data using Matplotlib. The package includes helper functions for creating grids that can represent various scenarios, variables, and time snapshots.

## Features

- **Grid Layout Helpers**: 
  - `make_grid_rows`: Create a grid where each row represents a scenario or variable, with columns for time snapshots.
  - `make_grid_columns`: Create a grid where each column represents a variable, with rows for scenarios or times.

- **Modular Design**: The package is designed to be modular, allowing for easy extension and customization.

## Installation

To install the package, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd flexplot-grid
pip install -r requirements.txt
```

## Usage

### Creating Grids

You can create grids using the provided functions. Here’s a quick example:

```python
from flexplot_grid import make_grid_rows, make_grid_columns

# Example usage of make_grid_rows
make_grid_rows(scenarios, times, mode, variables, scenario_configs, plotting_function, variable_configs, figsize)

# Example usage of make_grid_columns
make_grid_columns(variable_names, scenarios_or_times, mode, fixed_time_or_scenario, plotting_function, variable_configs, figsize)
```

### Example

Refer to the `examples/demo.py` file for a complete example of how to use the grid layout helpers in practice.

## Testing

To run the tests for the layout functions, including the new grid helpers, navigate to the `tests` directory and run:

```bash
pytest test_layout.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.