import numpy as np
import pytest

from imml.data_loader import DeepMFDataset

try:
    import torch
    torch_installed = True
except ImportError:
    torch_installed = False


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    X = torch.from_numpy(X)
    return X

def test_pytorch_not_installed(sample_data):
    if torch_installed:
        DeepMFDataset(X=sample_data)
    else:
        with pytest.raises(ImportError, match="torch and lightning needs to be installed."):
            DeepMFDataset(X=sample_data)

def test_default_params(sample_data):
    if torch_installed:
        dataset = DeepMFDataset(X=sample_data)
        assert len(dataset) == len(sample_data)

def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid X."):
        DeepMFDataset(X="invalid_input")

def test_getitem(sample_data):
    if torch_installed:
        dataset = DeepMFDataset(X=sample_data)
        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert sample[1].shape == (sample_data.shape[1],)


if __name__ == "__main__":
    pytest.main()