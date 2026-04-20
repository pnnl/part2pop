from matplotlib import pyplot as plt

def plot_lines(variable, populations, var_cfg=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for pop in populations:
        x, y = compute_variable(pop, variable, var_cfg)
        ax.plot(x, y, label=pop.label)

    return ax.get_lines(), ax.get_legend_handles_labels()

def compute_variable(population, variable, var_cfg):
    # Placeholder for actual computation logic
    x = population.get_variable_x(variable)
    y = population.get_variable_y(variable)
    return x, y

def plot_scatter(variable, populations, var_cfg=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for pop in populations:
        x, y = compute_variable(pop, variable, var_cfg)
        ax.scatter(x, y, label=pop.label)

    return ax.get_lines(), ax.get_legend_handles_labels()