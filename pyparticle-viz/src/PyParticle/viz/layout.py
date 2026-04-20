from matplotlib import gridspec

def make_grid(rows, cols, figsize=None):
    if figsize is None:
        figsize = (cols * 5, rows * 4)  # Default size for each subplot

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    axarr = []
    for r in range(rows):
        row_axes = []
        for c in range(cols):
            ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        axarr.append(row_axes)

    return fig, axarr