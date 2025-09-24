import numpy as np
import pandas as pd

from ..impute import get_observed_mod_indicator

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


def plot_missing_modality(Xs, figsize=None, sort: bool = True):
    r"""
    Plot modality missing.

    Parameters
    ----------
    Xs : list of array-likes, default=None
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities. If rus is provided, it will not be used.
    figsize : tuple, default=None
        Figure size (tuple) in inches.
    sort : bool, default=True
        If True, samples will be sort based on their available modalities.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure object.
    ax : `matplotlib.axes.Axes`
        Axes object.
    """
    if not isinstance(Xs, list):
        raise ValueError(f"Invalid Xs. It must be a list. A {type(Xs)} was passed.")
    if any(len(X) == 0 for X in Xs):
        raise ValueError("Invalid Xs. All elements must have at least one sample.")
    if len(set(len(X) for X in Xs)) > 1:
        raise ValueError("Invalid Xs. All elements must have the same number of samples.")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xlabel, ylabel = "Modality", "Samples"
    observed_view_indicator = get_observed_mod_indicator(Xs)
    observed_view_indicator = pd.DataFrame(observed_view_indicator)
    if sort:
        observed_view_indicator = observed_view_indicator.sort_values(list(range(len(Xs))))
    observed_view_indicator.columns = observed_view_indicator.columns + 1
    ax.pcolor(observed_view_indicator, cmap="binary", edgecolors="black", vmin=0., vmax=2.)
    ax.set_xticks(np.arange(0.5, len(observed_view_indicator.columns), 1), observed_view_indicator.columns)
    _ = ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    return fig, ax
