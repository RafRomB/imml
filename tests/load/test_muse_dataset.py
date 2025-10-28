import pytest
torch = pytest.importorskip("torch")
import importlib
import sys
from unittest.mock import patch
import numpy as np

from imml.load import MUSEDataset


@pytest.fixture
def sample_data():
    n_samples = 5
    n_mods = 3
    Xs = [torch.rand((n_samples, 10)) for _ in range(n_mods)]
    y = torch.randint(0, 2, (n_samples,), dtype=torch.float)
    return Xs, y


def test_deepmodule_not_installed(sample_data):
    Xs, y = sample_data
    MUSEDataset(Xs=Xs, y=y)
    with patch.dict(sys.modules, {"torch": None}):
        import imml.load.muse_dataset as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            MUSEDataset(Xs=Xs, y=y)
    importlib.reload(module_mock)


def test_default_params(sample_data):
    Xs, y = sample_data
    dataset = MUSEDataset(Xs=Xs, y=y)
    assert len(dataset) == len(y)
    assert hasattr(dataset, 'Xs')
    assert hasattr(dataset, 'y')
    assert hasattr(dataset, 'missing_mod_indicator')
    assert hasattr(dataset, 'y_indicator')
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 4
    assert len(sample[0]) == len(Xs)
    assert isinstance(sample[1], torch.Tensor)
    assert isinstance(sample[2], torch.Tensor)
    assert isinstance(sample[3], torch.Tensor)


def test_invalid_params():
    n_samples = 5
    Xs = [torch.rand((n_samples, 10)) for _ in range(3)]
    y = torch.randint(0, 2, (n_samples,), dtype=torch.float)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs="not_a_list", y=y)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs=[], y=y)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs=[torch.rand((n_samples, 10)), torch.rand((0, 10))], y=y)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs=[torch.rand((n_samples, 10)), torch.rand((n_samples+1, 10))], y=y)
    with pytest.raises(ValueError, match="Invalid y."):
        MUSEDataset(Xs=Xs, y=None)
    with pytest.raises(ValueError, match="Invalid y."):
        MUSEDataset(Xs=Xs, y=torch.randint(0, 2, (n_samples+1,)))


def test_getitem(sample_data):
    Xs, y = sample_data
    dataset = MUSEDataset(Xs=Xs, y=y)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, tuple)
        assert len(sample) == 4
        Xs_idx = sample[0]
        assert len(Xs_idx) == len(Xs)
        for j, X_idx in enumerate(Xs_idx):
            assert X_idx.shape == (Xs[j].shape[1],)
            assert torch.allclose(X_idx, Xs[j][i])
        y_idx = sample[1]
        assert isinstance(y_idx, torch.Tensor)
        assert y_idx.item() == y[i].item()
        observed_mod_indicator_idx = sample[2]
        assert isinstance(observed_mod_indicator_idx, torch.Tensor)
        y_indicator_idx = sample[3]
        assert isinstance(y_indicator_idx, torch.Tensor)


def test_missing_values(sample_data):
    Xs, y = sample_data
    Xs[0][0, :] = torch.nan
    Xs[1][1, :] = torch.nan
    dataset = MUSEDataset(Xs=Xs, y=y)
    sample = dataset[0]
    assert sample[2][0].item()
    assert not sample[2][1].item()
    assert not sample[2][2].item()
    sample = dataset[1]
    assert not sample[2][0].item()
    assert sample[2][1].item()
    assert not sample[2][2].item()
    sample = dataset[2]
    assert sample[3].item()


if __name__ == "__main__":
    pytest.main()