import importlib
import sys
from unittest.mock import patch
import numpy as np
import pytest

from imml.load import MRGCNDataset

try:
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    Xs = [X[:, :3], X[:, 3:5], X[:, 5:]]
    if deepmodule_installed:
        Xs = [torch.from_numpy(X) for X in Xs]
    return Xs


def test_deepmodule_not_installed(sample_data):
    if deepmodule_installed:
        MRGCNDataset(Xs=sample_data)
        with patch.dict(sys.modules, {"torch": None}):
            import imml.load.mrgcn_dataset as module_mock
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'Deep' needs to be installed."):
                MRGCNDataset(Xs=sample_data)
        importlib.reload(module_mock)
    else:
        with pytest.raises(ImportError, match="Module 'Deep' needs to be installed."):
            MRGCNDataset(Xs=sample_data)


@pytest.mark.skipif(not deepmodule_installed, reason="Module 'Deep' needs to be installed.")
def test_default_params(sample_data):
    dataset = MRGCNDataset(sample_data)
    assert len(dataset) == len(sample_data[0])
    assert dataset.transform is None
    assert hasattr(dataset, 'Xs')
    assert len(dataset.Xs) == len(sample_data)


@pytest.mark.skipif(not deepmodule_installed, reason="Module 'Deep' needs to be installed.")
def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid Xs."):
        MRGCNDataset(Xs="invalid_input")
    with pytest.raises(ValueError, match="Invalid Xs."):
        MRGCNDataset(Xs=[])
    if deepmodule_installed:
        with pytest.raises(ValueError, match="Invalid Xs."):
            MRGCNDataset(Xs=[torch.rand((5, 10)), torch.rand((0, 10))])
    if deepmodule_installed:
        with pytest.raises(ValueError, match="Invalid Xs."):
            MRGCNDataset(Xs=[torch.rand((5, 10)), torch.rand((6, 10))])


@pytest.mark.skipif(not deepmodule_installed, reason="Module 'Deep' needs to be installed.")
def test_getitem(sample_data):
    dataset = MRGCNDataset(sample_data)
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == len(sample_data)
    for s, X in zip(sample, sample_data):
        assert s.shape == (X.shape[1],)
    samples = [dataset[i] for i in range(3)]
    assert len(samples) == 3
    for sample in samples:
        assert isinstance(sample, tuple)
        assert len(sample) == len(sample_data)


@pytest.mark.skipif(not deepmodule_installed, reason="Module 'Deep' needs to be installed.")
def test_transform(sample_data):
    dataset = MRGCNDataset(sample_data,
                           transform=[lambda x: x for _ in range(len(sample_data))])
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == len(sample_data)


if __name__ == "__main__":
    pytest.main()
