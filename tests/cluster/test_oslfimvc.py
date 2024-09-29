from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import OSLFIMVC
from tests import mock_test

try:
    import oct2py
    oct2py_installed = True
except ImportError:
    oct2py_installed = False
estimator = OSLFIMVC

@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    X = pd.DataFrame(X, index=list(ascii_lowercase)[:len(X)], columns= [f"feature{i}" for i in range(X.shape[1])])
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_oct2py_not_installed(monkeypatch):
    if oct2py_installed:
        estimator(engine="matlab")
        mock_test()
        with pytest.raises(ModuleNotFoundError, match="Oct2Py needs to be installed to use matlab engine."):
            estimator(engine="matlab")
    else:
        with pytest.raises(ModuleNotFoundError, match="Oct2Py needs to be installed to use matlab engine."):
            estimator(engine="matlab")

@pytest.mark.skipif(not oct2py_installed, reason="Oct2py is not installed.")
def test_default_params(sample_data):
    model = estimator(random_state=42)
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
        assert model.embedding_.shape == (n_samples, model.n_clusters)
        assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
        assert len(model.beta_) == len(Xs)
        assert model.n_iter_ > 0

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)

@pytest.mark.skipif(not oct2py_installed, reason="Oct2py is not installed.")
def test_fit_predict(sample_data):
    n_clusters = 3
    model = estimator(n_clusters=n_clusters, random_state=42)
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
        assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
        assert len(model.beta_) == len(Xs)
        assert model.n_iter_ > 0

@pytest.mark.skipif(not oct2py_installed, reason="Oct2py is not installed.")
def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = estimator(n_clusters=n_clusters, random_state=42)
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
        assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
        assert len(model.beta_) == len(Xs)
        assert model.n_iter_ > 0


if __name__ == "__main__":
    pytest.main()