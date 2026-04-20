from matplotlib import gridspec

def create_scenario_grid(num_scenarios, num_times, figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_scenarios, num_times, figure=fig)
    return fig, gs

def create_variable_grid(num_variables, num_times, figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_variables, num_times, figure=fig)
    return fig, gs

def create_scenario_variable_grid(num_scenarios, num_variables, figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_scenarios, num_variables, figure=fig)
    return fig, gs

def create_time_variable_grid(num_times, num_variables, figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_times, num_variables, figure=fig)
    return fig, gs