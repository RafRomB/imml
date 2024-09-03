import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.decomposition import jNMF

try:
    rpy2_installed = True
except ImportError:
    rpy2_installed = False


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((50, 270))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :100], X.iloc[:, 100:190], X.iloc[:, 190]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_rpy2_not_installed():
    if rpy2_installed:
        jNMF(engine="r")
    else:
        with pytest.raises(ModuleNotFoundError, match="rpy2 needs to be installed to use matlab engine."):
            jNMF(engine="r")

def test_random_state(sample_data):
    if rpy2_installed:
        jNMF()

def test_default_params(sample_data):
    if rpy2_installed:
        transformer = jNMF(random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'H_')
            assert hasattr(transformer, 'reconstruction_err_')
            assert hasattr(transformer, 'observed_reconstruction_err_')
            assert hasattr(transformer, 'missing_reconstruction_err_')
            assert hasattr(transformer, 'relchange_')

def test_fit(sample_data):
    n_components = 5
    if rpy2_installed:
        transformer = jNMF(n_components=n_components, max_iter=10, random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'H_')
            assert hasattr(transformer, 'reconstruction_err_')
            assert hasattr(transformer, 'observed_reconstruction_err_')
            assert hasattr(transformer, 'missing_reconstruction_err_')
            assert hasattr(transformer, 'relchange_')

def test_transform(sample_data):
    n_components = 5
    if rpy2_installed:
        transformer = jNMF(n_components=n_components, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            transformer.fit(Xs)
            transformed_X = transformer.transform(Xs)
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)

def test_set_output(sample_data):
    if rpy2_installed:
        transformer = jNMF(n_components=5, random_state=42).set_output(transform="pandas")
        assert transformer.transform_ == "pandas"
        for Xs in sample_data:
            transformed_X = transformer.fit_transform(Xs)
            assert isinstance(transformed_X, pd.DataFrame)

def test_missing_values_handling(sample_data):
    n_components = 5
    if rpy2_installed:
        transformer = jNMF(n_components=n_components, random_state=42).set_output(transform="pandas")
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            transformed_X = transformer.fit_transform(Xs)
            assert not np.isnan(transformed_X).any().any()
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid engine"):
        jNMF(engine="invalid")


if __name__ == "__main__":
    pytest.main()