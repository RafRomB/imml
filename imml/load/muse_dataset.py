try:
    import lightning.pytorch as pl
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

TorchDatasetBase = torch.utils.data.Dataset if deepmodule_installed else object


class MUSEDataset(TorchDatasetBase):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `M3Care`.

    Parameters
    ----------
    Xs : list of array-likes
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    transform : callable, defult=None
        A function or transformation to apply to each sample in the dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.load import MRGCNDataset
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> train_data = MRGCNDataset(Xs=Xs)
    """

    def __init__(self, Xs, y, observed_mod_indicator, y_indicator):
        self.Xs = Xs
        self.y = y
        self.observed_mod_indicator = observed_mod_indicator
        self.y_indicator = y_indicator


    def __len__(self):
        return len(self.observed_mod_indicator)


    def __getitem__(self, idx):
        Xs_idx = [X[idx] for X in self.Xs]
        y_idx = self.y[idx]
        observed_mod_indicator_idx = self.observed_mod_indicator[idx]
        y_indicator_idx = self.y_indicator[idx]
        return Xs_idx, y_idx, observed_mod_indicator_idx, y_indicator_idx