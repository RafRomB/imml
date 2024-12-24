import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.decomposition import JNMF

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    rpy2_installed = True
except ImportError:
    rpy2_installed = False
    rpy2_module_error = "rpy2 needs to be installed to use r engine."


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_rpy2_not_installed():
    if rpy2_installed:
        JNMF(engine="r")
    else:
        with pytest.raises(ImportError, match="rpy2 needs to be installed to use matlab engine."):
            JNMF(engine="r")

def test_random_state(sample_data):
    if rpy2_installed:
        JNMF()

def test_default_params(sample_data):
    if rpy2_installed:
        transformer = JNMF(random_state=42)
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
        transformer = JNMF(n_components=n_components, max_iter=10, random_state=42)
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
        transformer = JNMF(n_components=n_components, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            transformer.fit(Xs)
            transformed_X = transformer.transform(Xs)
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)

def test_missing_values_handling(sample_data):
    n_components = 5
    if rpy2_installed:
        transformer = JNMF(n_components=n_components, random_state=42)
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
        JNMF(engine="invalid")


if __name__ == "__main__":
    pytest.main()