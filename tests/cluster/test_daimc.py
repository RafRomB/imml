from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import DAIMC


@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((20, 3)),
                      index=list(ascii_lowercase)[:20],
                      columns=['feature1', 'feature2', 'feature3'])
    X2 = pd.DataFrame(np.random.default_rng(42).random((20, 2)),
                      index=list(ascii_lowercase)[:20],
                      columns=['feature4', 'feature5'])
    X3 = pd.DataFrame(np.random.default_rng(42).random((20, 5)),
                      index=list(ascii_lowercase)[:20],
                      columns=['feature6', 'feature7', 'feature8', 'feature9', 'feature10'])
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_parameters(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    model = DAIMC(random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape[0] == n_samples
        assert len(model.U_) == len(Xs)
        assert len(model.B_) == len(Xs)

def test_custom_parameters(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_clusters = 3
    model = DAIMC(n_clusters=n_clusters, alpha=0.5, beta=0.5, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert not np.isnan(labels).any()
        assert len(model.U_) == len(Xs)
        assert len(model.B_) == len(Xs)
        assert model.U_[0].shape == (Xs[0].shape[1], n_clusters)
        assert model.B_[0].shape == (Xs[0].shape[1], n_clusters)

def test_invalid_engine(sample_data):
    with pytest.raises(ValueError, match="Only engine=='matlab' is currently supported."):
        DAIMC(engine='invalid')
    with pytest.raises(ValueError, match="Only engine=='matlab' is currently supported."):
        model = DAIMC()
        Xs_pandas, Xs_numpy = sample_data
        model.engine = 'invalid'
        model.fit(Xs_pandas)

def test_fit_predict(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_clusters = 3
    model = DAIMC(n_clusters=n_clusters, random_state=42)
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
        assert len(model.U_) == len(Xs)
        assert len(model.B_) == len(Xs)
        assert model.U_[0].shape == (Xs[0].shape[1], n_clusters)
        assert model.B_[0].shape == (Xs[0].shape[1], n_clusters)

def test_missing_values_handling(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_clusters = 2
    model = DAIMC(n_clusters=n_clusters, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert len(model.U_) == len(Xs)
        assert len(model.B_) == len(Xs)
        assert model.U_[0].shape == (Xs[0].shape[1], n_clusters)
        assert model.B_[0].shape == (Xs[0].shape[1], n_clusters)

if __name__ == "__main__":
    pytest.main()