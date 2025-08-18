import networkx as nx
import numpy as np
from sklearn.utils import check_symmetric
from snf.compute import _find_dominate_set

try:
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

TorchDatasetBase = torch.utils.data.Dataset if deepmodule_installed else object


class IntegrAODataset(TorchDatasetBase):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `M3Care`.

    Parameters
    ----------
    Xs : list of array-likes
        - Xs length: n_mods

        A list of different modalities.
    y : array-like of shape (n_samples,)
        Target vector relative to X.
    observed_mod_indicator: array-like of shape (n_samples, n_mods)
        Boolean array-like indicating observed modalities for each sample.

    Returns
    -------
    Xs_idx: list of array-likes
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
    >>> observed_mod_indicator = torch.from_numpy(get_observed_mod_indicator(Xs).values)
    >>> y = torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float()
    >>> train_data = M3CareDataset(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y)
    """

    def __init__(self, Xs, neighbor_size : int, networks : list):
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

        self.Xs = []
        self.edge_index = []
        self.indexes = []
        for X, network in zip(Xs, networks):
            idxs = X.dropna(axis=0).index
            idxs = torch.from_numpy(idxs.values).long()
            X = torch.from_numpy(X.values).type(torch.float32)
            self.Xs.append(X)
            self.indexes.append(idxs)
            neighbor_size = min(int(neighbor_size), network.shape[0])

            network = _find_dominate_set(network, K=neighbor_size)
            network = check_symmetric(network, raise_warning=False)
            network[network > 0.0] = 1.0
            G = nx.from_numpy_array(network)

            adj = nx.to_scipy_sparse_array(G).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            self.edge_index.append(edge_index)


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        Xs = [X[idx] for X in self.Xs]
        indexes = [idx for idx in self.indexes]
        edge_index = self.edge_index
        return Xs, edge_index, indexes, idx
