from string import ascii_lowercase
import numpy as np
import pandas as pd
import pytest

from imml.load import MRGCNDataset

try:
    import torch
    torch_installed = True
except ImportError:
    torch_installed = False


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    Xs = [X[:, :3], X[:, 3:5], X[:, 5:]]
    if torch_installed:
        Xs = [torch.from_numpy(X) for X in Xs]
    return Xs

def test_pytorch_not_installed(sample_data):
    if torch_installed:
        MRGCNDataset(Xs=sample_data)
    else:
        with pytest.raises(ImportError, match="torch and lightning needs to be installed."):
            MRGCNDataset(Xs=sample_data)

def test_default_params(sample_data):
    if torch_installed:
        dataset = MRGCNDataset(sample_data)
        assert len(dataset) == len(sample_data[0])

def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid Xs."):
        MRGCNDataset(Xs="invalid_input")

def test_getitem(sample_data):
    if torch_installed:
        dataset = MRGCNDataset(sample_data)
        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == len(sample_data)
        for s, X in zip(sample, sample_data):
            assert s.shape == (X.shape[1],)

if __name__ == "__main__":
    pytest.main()