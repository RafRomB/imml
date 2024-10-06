from string import ascii_lowercase
from unittest import mock

import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import OMVC


try:
    import oct2py
    oct2py_installed = True
except ImportError:
    oct2py_installed = False
estimator = OMVC

@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((8, 5))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :2], X.iloc[:, 2:4], X.iloc[:, 4:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_oct2py_not_installed(monkeypatch):
    if oct2py_installed:
        estimator(engine="matlab")
        with mock.patch("imvc.cluster.omvc.oct2py_installed", False):
            with mock.patch("imvc.cluster.omvc.oct2py_module_error",
                            "Oct2Py needs to be installed to use matlab engine."):
                with pytest.raises(ImportError, match="Oct2Py needs to be installed to use matlab engine."):
                    estimator(engine="matlab")
    else:
        with pytest.raises(ImportError, match="Oct2Py needs to be installed to use matlab engine."):
            estimator(engine="matlab")

def test_default_params(sample_data):
    model = OMVC(random_state=42)
    if oct2py_installed:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == model.n_clusters
            assert min(labels) == 0
            assert max(labels) == (model.n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape[0] == n_samples
            assert len(model.U_) == len(Xs)
            assert len(model.V_) == len(Xs)
            assert model.n_iter_ > 0

def test_invalid_params(sample_data):
    estimator = OMVC
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)

def test_fit_predict(sample_data):
    n_clusters = 3
    model = OMVC(n_clusters=n_clusters, random_state=42)
    if oct2py_installed:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert len(model.U_) == len(Xs)
            assert len(model.V_) == len(Xs)
            assert model.U_[0].shape == (n_samples, n_clusters)
            assert model.V_[0].shape == (Xs[0].shape[1], n_clusters)
            assert model.n_iter_ > 0

def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = OMVC(n_clusters=n_clusters, random_state=42)
    if oct2py_installed:
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert len(model.U_) == len(Xs)
            assert len(model.V_) == len(Xs)
            assert model.U_[0].shape == (n_samples, n_clusters)
            assert model.V_[0].shape == (Xs[0].shape[1], n_clusters)
            assert model.n_iter_ > 0

if __name__ == "__main__":
    pytest.main()