# License: BSD-3-Clause
import numpy as np
import pandas as pd

from ..impute import get_observed_mod_indicator

try:
    import lightning.pytorch as pl
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

Dataset = torch.utils.data.Dataset if deepmodule_installed else object


class M3CareDataset(Dataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `M3Care`.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods

        A list of different modalities.
    y : array-like of shape (n_samples,)
        Target vector relative to X.

    Returns
    -------
    Xs_idx: list of array-likes objects
        - Xs length: n_mods

        A list of different modalities for one sample.
    y_idx: array-like of shape (n_samples,)
        Target vector relative to the sample.
    observed_mod_indicator: array-like of shape (1, n_mods)
        Boolean array-like indicating observed modalities for the sample.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.load import M3CareDataset
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = [torch.from_numpy(X.values).float() for X in Xs]
    >>> y = torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float()
    >>> train_data = M3CareDataset(Xs=Xs, y=y)
    """

    def __init__(self, Xs, y):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list. A {type(Xs)} was passed.")
        if len(Xs) < 2:
            raise ValueError(f"Invalid Xs. It must have at least two modalities. Got {len(Xs)} modalities.")
        if any(len(X) == 0 for X in Xs):
            raise ValueError("Invalid Xs. All elements must have at least one sample.")
        if len(set(len(X) for X in Xs)) > 1:
            raise ValueError("Invalid Xs. All elements must have the same number of samples.")
        if y is None:
            raise ValueError("Invalid y. It cannot be None.")
        if len(y) != len(Xs[0]):
            raise ValueError(f"Invalid y. It must have the same length as each element in Xs. Got {len(y)} vs {len(Xs[0])}")

        observed_mod_indicator = get_observed_mod_indicator(Xs)

        self.Xs = Xs
        self.y = y
        self.observed_mod_indicator = observed_mod_indicator


    def __len__(self):
        return len(self.observed_mod_indicator)


    def __getitem__(self, idx):
        Xs_idx = [X[idx] for X in self.Xs]
        y_idx = self.y[idx]
        observed_mod_indicator_idx = self.observed_mod_indicator[idx]
        return Xs_idx, y_idx, observed_mod_indicator_idx
