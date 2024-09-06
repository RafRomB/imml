from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import PIMVC

try:
    import oct2py
    oct2py_installed = True
except ImportError:
    oct2py_installed = False

@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 50))
    X = pd.DataFrame(X, index=list(ascii_lowercase)[:len(X)], columns= [f"feature{i}" for i in range(X.shape[1])])
    X1, X2, X3 = X.iloc[:, :15], X.iloc[:, 15:32], X.iloc[:, 32:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_oct2py_not_installed():
    if oct2py_installed:
        PIMVC(engine="matlab")
    else:
        with pytest.raises(ModuleNotFoundError, match="Oct2Py needs to be installed to use matlab engine."):
            PIMVC(engine="matlab")

def test_default_params(sample_data):
    model = PIMVC(random_state=42)
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
            assert model.n_iter_ > 0

def test_invalid_params(sample_data):
    estimator = PIMVC
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)
    with pytest.raises(ValueError, match="Invalid lamb."):
        PIMVC(lamb=-1)
    with pytest.raises(ValueError, match="Invalid k."):
        PIMVC(k=-1)
    if oct2py_installed:
        with pytest.raises(ValueError, match="n_clusters should be smaller or equal to the smallest n_features_i."):
            model = PIMVC(n_clusters=sample_data[0][0].shape[1] + 1)
            model.fit(sample_data[0])

def test_fit_predict(sample_data):
    n_clusters = 3
    model = PIMVC(n_clusters=n_clusters, random_state=42)
    if oct2py_installed:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert model.n_iter_ > 0

def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = PIMVC(n_clusters=n_clusters, random_state=42)
    if oct2py_installed:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            labels = model.fit_predict(Xs)
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert model.n_iter_ > 0

if __name__ == "__main__":
    pytest.main()