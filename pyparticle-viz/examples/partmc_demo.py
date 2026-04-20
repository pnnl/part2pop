from pathlib import Path
import json
import matplotlib.pyplot as plt

from PyParticle.population import build_population
from PyParticle.viz import make_grid, plot_lines, format_axes, add_legend

def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "configs" / "partmc_example.json"
    out_png = repo_root / "partmc_demo.png"
    cfg = json.load(open(cfg_path))

    # Define scenarios and variables for demonstration
    scenarios = ["Scenario 1", "Scenario 2"]
    variables = ["dNdlnD", "other_variable"]
    times = [1, 12, 24]

    # Create a grid for different scenarios (same variable)
    fig1, axarr1 = make_grid(len(scenarios), len(times), figsize=(12, 8))
    for i, scenario in enumerate(scenarios):
        for j, t in enumerate(times):
            cfg_t = dict(cfg)
            cfg_t['timestep'] = t
            pop = build_population(cfg_t)
            ax = axarr1[i, j]
            line, labs = plot_lines("dNdlnD", (pop,), var_cfg=None, ax=ax)
            format_axes(ax, xlabel=labs[0], ylabel=labs[1], title=f"{scenario}, t={t}")
            ax.set_xscale("log")
            add_legend(ax)
    fig1.suptitle("Different Scenarios Over Time")
    fig1.savefig(out_png)

    # Create a grid for different variables (same scenario)
    fig2, axarr2 = make_grid(len(variables), len(times), figsize=(12, 8))
    for i, variable in enumerate(variables):
        for j, t in enumerate(times):
            cfg_t = dict(cfg)
            cfg_t['timestep'] = t
            pop = build_population(cfg_t)
            ax = axarr2[i, j]
            line, labs = plot_lines(variable, (pop,), var_cfg=None, ax=ax)
            format_axes(ax, xlabel=labs[0], ylabel=labs[1], title=f"{variable}, t={t}")
            ax.set_xscale("log")
            add_legend(ax)
    fig2.suptitle("Different Variables Over Time")
    fig2.savefig(repo_root / "partmc_demo_variables.png")

if __name__ == "__main__":
    main()