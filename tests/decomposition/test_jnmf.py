import importlib
import sys
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.decomposition import JNMF

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    rmodule_installed = True
except ImportError:
    rmodule_installed = False
    rmodule_error = "Module 'r' needs to be installed to use r engine."

nnTensor_installed = False
if rmodule_installed:
    rbase = importr("base")
    try:
        nnTensor = importr("nnTensor")
        nnTensor_installed = True
    except PackageNotInstalledError:
        pass
estimator = JNMF


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

@pytest.mark.skipif(not nnTensor_installed, reason="nnTensor is not installed.")
def test_rmodule_installed():
    if rmodule_installed:
        if nnTensor_installed:
            estimator(engine="r")
        else:
            with pytest.raises(ImportError, match="nnTensor needs to be installed in R to use r engine."):
                estimator(engine="r")


def test_random_state(sample_data):
    if rmodule_installed:
        estimator()

def test_default_params(sample_data):
    if rmodule_installed:
        transformer = estimator(random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'H_')
            assert hasattr(transformer, 'reconstruction_err_')
            assert hasattr(transformer, 'observed_reconstruction_err_')
            assert hasattr(transformer, 'missing_reconstruction_err_')
            assert hasattr(transformer, 'relchange_')

def test_fit(sample_data):
    n_components = 5
    if rmodule_installed:
        transformer = estimator(n_components=n_components, max_iter=10, random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'H_')
            assert hasattr(transformer, 'reconstruction_err_')
            assert hasattr(transformer, 'observed_reconstruction_err_')
            assert hasattr(transformer, 'missing_reconstruction_err_')
            assert hasattr(transformer, 'relchange_')

def test_transform(sample_data):
    n_components = 5
    if rmodule_installed:
        transformer = estimator(n_components=n_components, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            transformer.fit(Xs)
            transformed_X = transformer.transform(Xs)
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)

def test_missing_values_handling(sample_data):
    n_components = 5
    if rmodule_installed:
        transformer = estimator(n_components=n_components, random_state=42)
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
        estimator(engine="invalid")


if __name__ == "__main__":
    pytest.main()