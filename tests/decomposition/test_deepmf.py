import pandas as pd
import pytest
import numpy as np

from imvc.decomposition import DeepMF
from imvc.data_loader import DeepMFDataset

try:
    import torch
    from torch.utils.data import DataLoader
    from lightning import Trainer
    torch_installed = True
except ImportError:
    torch_installed = False


@pytest.fixture
def sample_data():
    if torch_installed:
        X = np.random.default_rng(42).random((25, 100), dtype=np.float32).T
        X = torch.from_numpy(X)
        return X

def test_pytorch_not_installed(sample_data):
    if torch_installed:
        DeepMF(X=sample_data)
    else:
        with pytest.raises(ModuleNotFoundError, match="torch and lightning needs to be installed."):
            DeepMF(X=sample_data)

def test_default_params(sample_data):
    n_features, n_samples = sample_data.shape
    train_data = DeepMFDataset(X=sample_data)
    train_dataloader = DataLoader(dataset=train_data, batch_size=n_features, shuffle=True)
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    transformer = DeepMF(X=sample_data)
    trainer.fit(transformer, train_dataloader)
    transformed_X = transformer.transform(sample_data)
    assert transformed_X.shape == (n_samples, transformer.n_components)
    assert transformer.U_.shape == (n_features, transformer.n_components)
    assert transformer.V_.shape == (transformer.n_components, n_samples)
    assert not torch.isnan(transformed_X).any().any()
    assert not torch.isnan(transformer.V_).any().any()
    assert not torch.isnan(transformer.V_).any().any()


def test_init(sample_data):
    n_components = 5
    transformer = DeepMF(n_components=n_components, X=sample_data)
    assert transformer.n_components == n_components

def test_set_output(sample_data):
    n_components = 5
    n_features, n_samples = sample_data.shape
    train_data = DeepMFDataset(X=sample_data)
    train_dataloader = DataLoader(dataset=train_data, batch_size=n_features, shuffle=True)
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    transformer = DeepMF(n_components=n_components, X=sample_data).set_output(transform="pandas")
    assert transformer.transform_ == "pandas"
    trainer.fit(transformer, train_dataloader)
    transformed_X = transformer.transform(sample_data)
    assert isinstance(transformed_X, pd.DataFrame)

def test_missing_values_handling(sample_data):
    n_components = 5
    sample_data[:, :5] = np.nan
    n_features, n_samples = sample_data.shape
    train_data = DeepMFDataset(X=sample_data)
    train_dataloader = DataLoader(dataset=train_data, batch_size=n_features, shuffle=True)
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    transformer = DeepMF(n_components=n_components, X=sample_data)
    trainer.fit(transformer, train_dataloader)
    transformed_X = transformer.transform(sample_data)
    assert transformed_X.shape == (n_samples, n_components)
    assert transformer.U_.shape == (n_features, n_components)
    assert transformer.V_.shape == (n_components, n_samples)
    assert not torch.isnan(transformer.V_).any().any()
    assert not torch.isnan(transformer.V_).any().any()

def test_invalid_params(sample_data):
    estimator = DeepMF
    with pytest.raises(ValueError, match="Invalid n_components."):
        estimator(n_components='invalid', X=sample_data)
    with pytest.raises(ValueError, match="Invalid n_components."):
        estimator(n_components=0, X=sample_data)
    with pytest.raises(ValueError, match="Invalid X."):
        estimator(X="invalid")


if __name__ == "__main__":
    pytest.main()