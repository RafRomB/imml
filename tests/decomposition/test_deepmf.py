import pytest
import numpy as np
import torch

from imvc.ampute import Amputer
from imvc.decomposition import MOFA


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    X = torch.from_numpy(X)
    return X

def test_default_params(sample_data):
    transformer = MOFA()
    for Xs in sample_data:
        transformer.fit(Xs)
        assert hasattr(transformer, 'mofa_')
        assert hasattr(transformer, 'weights_')

def test_init(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    assert transformer.n_components == n_components
    assert transformer.random_state == 42

def test_fit(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        transformer.fit(Xs)
        assert hasattr(transformer, 'mofa_')
        assert hasattr(transformer, 'weights_')

def test_transform(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        transformer.fit(Xs)
        transformed_X = transformer.transform(Xs)
        assert len(transformed_X) == len(Xs)
        assert transformed_X[0].shape == (n_samples, n_components)
        assert len(transformer.weights_) == len(Xs)
        assert transformer.weights_[0].shape == (Xs[0].shape[1], n_components)

def test_fit_transform(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        transformed_X = transformer.fit_transform(Xs)
        assert not np.isnan(transformed_X).any().any()
        assert transformed_X.shape == (n_samples, n_components)
        assert len(transformer.weights_) == len(Xs)
        assert transformer.weights_[0].shape == (Xs[0].shape[1], n_components)

def test_set_output(sample_data):
    transformer = MOFA(n_components=5, random_state=42, verbose=True).set_output(transform="pandas")
    assert transformer.transform_ == "pandas"
    for Xs in sample_data:
        transformed_X = transformer.fit_transform(Xs)
        assert isinstance(transformed_X, pd.DataFrame)

def test_missing_values_handling(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42).set_output(transform="pandas")
    for Xs in sample_data:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        transformed_X = transformer.fit_transform(Xs)
        assert not np.isnan(transformed_X).any().any()
        assert transformed_X.shape == (n_samples, n_components)
        assert len(transformer.weights_) == len(Xs)
        assert transformer.weights_[0].shape == (Xs[0].shape[1], n_components)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid n_components"):
        MOFA(n_components=0)


if __name__ == "__main__":
    pytest.main()