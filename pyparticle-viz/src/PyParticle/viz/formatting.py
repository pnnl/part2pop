from matplotlib import legend

def format_axes(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def add_legend(ax, loc='best'):
    ax.legend(loc=loc)