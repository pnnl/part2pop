# Usage of PyParticle.viz

The `PyParticle.viz` package provides a set of utilities for visualizing particle population data. This document outlines how to use the various features of the package, including creating figure grids for different scenarios and variables.

## Creating Figure Grids

### Standard Figure Grid Types

1. **Row-wise Grids**:
   - Each row can represent either:
     - **Different Scenarios**: The same variables are plotted across different scenarios, with each column representing a different time snapshot.
     - **Different Variables**: The same scenario is used, but different variables are plotted across the rows, with each column representing a different time snapshot.

   Example usage:
   ```python
   from PyParticle.viz import make_grid, plot_lines

   # Create a grid for different scenarios
   fig, axarr = make_grid(num_rows, num_columns)
   for scenario in scenarios:
       for time in times:
           pop = build_population(scenario, time)
           plot_lines(variable, pop, ax=axarr[row, column])
   ```

2. **Column-wise Grids**:
   - Each column can represent either:
     - **Different Variables**: Different scenarios are plotted in the rows for a single time snapshot.
     - **Different Times**: The same scenario is used, but different times are plotted across the rows for a single variable.

   Example usage:
   ```python
   from PyParticle.viz import make_grid, plot_lines

   # Create a grid for different variables
   fig, axarr = make_grid(num_rows, num_columns)
   for variable in variables:
       for scenario in scenarios:
           pop = build_population(scenario, time)
           plot_lines(variable, pop, ax=axarr[row, column])
   ```

## Plotting Variables

The `plot_lines` function allows you to visualize different variables from the particle population. You can customize the appearance of the plots using the styling utilities provided in the package.

### Example of Plotting

```python
from PyParticle.viz import plot_lines, format_axes

# Plotting a specific variable
line, labels = plot_lines("dNdlnD", populations, ax=ax)
format_axes(ax, xlabel='Size', ylabel='Density', title='Density vs Size')
```

## Conclusion

The `PyParticle.viz` package is designed to facilitate the visualization of particle population data through flexible grid layouts and customizable plotting functions. Use the examples provided to create your own visualizations tailored to your specific scenarios and variables.