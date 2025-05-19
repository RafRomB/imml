try:
    import torch
    DLBaseDataset = torch.utils.data.Dataset
    deep_installed = True
except ImportError:
    deep_installed = False
    deep_module_error = "Module 'Deep' needs to be installed."
    DLBaseDataset = object


class MRGCNDataset(DLBaseDataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `MRGCN`.

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

    def __init__(self, Xs: list, transform = None):
        if not deep_installed:
            raise ImportError(deep_module_error)
        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list of array-likes. A {type(Xs)} was passed.")

        self.Xs = Xs
        self.transform = transform


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        if self.transform is not None:
            Xs = [self.transform[X_idx](X[idx]) for X_idx ,X in enumerate(self.Xs)]
        else:
            Xs = [X[idx] for X in self.Xs]
        Xs = tuple(Xs)
        return Xs