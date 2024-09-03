from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import NEMO

#todo test engine
try:
    rpy2_installed = True
except ImportError:
    rpy2_installed = False


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X, index=list(ascii_lowercase)[:len(X)], columns= [f"feature{i}" for i in range(X.shape[1])])
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_rpy2_not_installed():
    if rpy2_installed:
        NEMO(engine="r")
    else:
        with pytest.raises(ModuleNotFoundError, match="rpy2 needs to be installed to use matlab engine."):
            NEMO(engine="r")

def test_default_params(sample_data):
    model = NEMO(random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters_
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        assert not np.isnan(model.embedding_).any().any()
        assert model.embedding_.shape == (n_samples, model.n_clusters_)
        assert model.affinity_matrix_.shape == (n_samples, n_samples)
        assert len(model.num_neighbors_) == len(Xs)
        assert model.num_neighbors_[0] > 0

def test_custom_parameters(sample_data):
    model = NEMO(n_clusters=list(range(2, 4)), num_neighbors=3, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters_
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters_ - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, model.n_clusters_)
        assert model.affinity_matrix_.shape == (n_samples, n_samples)
        assert len(model.num_neighbors_) == len(Xs)
        assert model.num_neighbors_[0] > 0
    NEMO(n_clusters=list(range(2, 4)), num_neighbors=[3, 3, 3], random_state=42).fit_predict(Xs)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid engine."):
        NEMO(engine='invalid')

def test_fit_predict(sample_data):
    n_clusters = 3
    model = NEMO(n_clusters=n_clusters, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(np.unique(labels)) == model.n_clusters_
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert not np.isnan(model.embedding_).any().any()
        assert model.embedding_.shape == (n_samples, model.n_clusters_)
        assert model.affinity_matrix_.shape == (n_samples, n_samples)
        assert len(model.num_neighbors_) == len(Xs)
        assert model.num_neighbors_[0] > 0

def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = NEMO(n_clusters=n_clusters, random_state=42)
    for Xs in sample_data:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(np.unique(labels)) == model.n_clusters_
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert not np.isnan(model.embedding_).any().any()
        assert model.embedding_.shape == (n_samples, model.n_clusters_)
        assert model.affinity_matrix_.shape == (n_samples, n_samples)
        assert len(model.num_neighbors_) == len(Xs)
        assert model.num_neighbors_[0] > 0

if __name__ == "__main__":
    pytest.main()