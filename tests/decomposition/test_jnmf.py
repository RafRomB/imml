import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.decomposition import jNMF

try:
    import rpy2
    RPY2_INSTALLED = True
except ImportError:
    RPY2_INSTALLED = False


@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((50, 100)))
    X2 = pd.DataFrame(np.random.default_rng(42).random((50, 90)))
    X3 = pd.DataFrame(np.random.default_rng(42).random((50, 80)))
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_jNMF_default(sample_data):
    if RPY2_INSTALLED:
        transformer = jNMF()
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'H_')
            assert hasattr(transformer, 'reconstruction_err_')
            assert hasattr(transformer, 'observed_reconstruction_err_')
            assert hasattr(transformer, 'missing_reconstruction_err_')
            assert hasattr(transformer, 'relchange_')

def test_jNMF_init(sample_data):
    n_components = 5
    if RPY2_INSTALLED:
        transformer = jNMF(n_components=n_components, max_iter=200, random_state=42)
        assert transformer.n_components == n_components
        assert transformer.max_iter == 200
        assert transformer.random_state == 42

def test_jNMF_fit(sample_data):
    n_components = 5
    if RPY2_INSTALLED:
        transformer = jNMF(n_components=n_components, max_iter=10, random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'H_')
            assert hasattr(transformer, 'reconstruction_err_')
            assert hasattr(transformer, 'observed_reconstruction_err_')
            assert hasattr(transformer, 'missing_reconstruction_err_')
            assert hasattr(transformer, 'relchange_')

def test_jNMF_transform(sample_data):
    n_components = 5
    if RPY2_INSTALLED:
        transformer = jNMF(n_components=n_components, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            transformer.fit(Xs)
            transformed_X = transformer.transform(Xs)
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)

def test_jNMF_set_output(sample_data):
    if RPY2_INSTALLED:
        transformer = jNMF(n_components=5, random_state=42).set_output(transform="pandas")
        assert transformer.transform_ == "pandas"
        for Xs in sample_data:
            transformed_X = transformer.fit_transform(Xs)
            assert isinstance(transformed_X, pd.DataFrame)

def test_missing_values_handling(sample_data):
    n_components = 5
    if RPY2_INSTALLED:
        transformer = jNMF(n_components=n_components, random_state=42).set_output(transform="pandas")
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            transformed_X = transformer.fit_transform(Xs)
            assert not np.isnan(transformed_X).any().any()
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)

def test_jNMF_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid engine"):
        jNMF(engine="invalid")
    with pytest.raises(ValueError, match="Invalid engine"):
        model = jNMF()
        model.engine = "invalid"
        model.fit(sample_data[0])


if __name__ == "__main__":
    pytest.main()