from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import MSNE


@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((25, 3)),
                      index=list(ascii_lowercase)[:25],
                      columns=['feature1', 'feature2', 'feature3'])
    X2 = pd.DataFrame(np.random.default_rng(42).random((25, 2)),
                      index=list(ascii_lowercase)[:25],
                      columns=['feature4', 'feature5'])
    X3 = pd.DataFrame(np.random.default_rng(42).random((25, 5)),
                      index=list(ascii_lowercase)[:25],
                      columns=['feature6', 'feature7', 'feature8', 'feature9', 'feature10'])
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_parameters(sample_data):
    model = MSNE(random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, model.embed_size)

def test_custom_parameters(sample_data):
    n_clusters = 3
    model = MSNE(n_clusters=n_clusters, embed_size=10, random_state=42, verbose=True)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, model.embed_size)

def test_invalid_parameters(sample_data):
    with pytest.raises(ValueError, match="n_clusters should be a positive value."):
        MSNE(n_clusters=-1)
    with pytest.raises(ValueError, match="k should be smaller than the number of samples."):
        MSNE(k= 1000, random_state=42).fit_predict(sample_data[0])

def test_fit_predict(sample_data):
    n_clusters = 3
    model = MSNE(n_clusters=n_clusters, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, model.embed_size)

def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = MSNE(n_clusters=n_clusters, random_state=42)
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
        assert model.embedding_.shape == (n_samples, model.embed_size)

if __name__ == "__main__":
    pytest.main()