from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import PIMVC


@pytest.fixture
def sample_data():
    n_samples = 20
    X1 = pd.DataFrame(np.random.default_rng(42).random((n_samples, 20)),
                      index=list(ascii_lowercase)[:n_samples])
    X2 = pd.DataFrame(np.random.default_rng(42).random((n_samples, 15)),
                      index=list(ascii_lowercase)[:n_samples])
    X3 = pd.DataFrame(np.random.default_rng(42).random((n_samples, 10)),
                      index=list(ascii_lowercase)[:n_samples])
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_parameters(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    model = PIMVC(random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        assert not np.isnan(labels).any()
        assert model.embedding_.shape[0] == n_samples
        assert model.n_iter_ > 0

def test_custom_parameters(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_clusters = 3
    model = PIMVC(n_clusters=n_clusters, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert model.n_iter_ > 0

def test_invalid_parameters(sample_data):
    with pytest.raises(ValueError, match="Only engine=='matlab' is currently supported."):
        PIMVC(engine='invalid')
    with pytest.raises(ValueError, match="Only engine=='matlab' is currently supported."):
        model = PIMVC()
        Xs_pandas, Xs_numpy = sample_data
        model.engine = 'invalid'
        model.fit(Xs_pandas)
    with pytest.raises(ValueError, match="lamb should be a positive value."):
        PIMVC(lamb=-1)
    with pytest.raises(ValueError, match="k should be a positive value."):
        PIMVC(k=-1)
    with pytest.raises(ValueError, match="n_clusters should be smaller or equal to the smallest n_features_i."):
        model = PIMVC(n_clusters=11)
        Xs_pandas, Xs_numpy = sample_data
        model.fit(Xs_pandas)

def test_fit_predict(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_clusters = 3
    model = PIMVC(n_clusters=n_clusters, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert model.n_iter_ > 0

def test_missing_values_handling(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_clusters = 2
    model = PIMVC(n_clusters=n_clusters, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        labels = model.fit_predict(Xs)
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert model.n_iter_ > 0

if __name__ == "__main__":
    pytest.main()