from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import OSLFIMVC

try:
    import oct2py
    OCT2PY_INSTALLED = True
except ImportError:
    OCT2PY_INSTALLED = False


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
    model = OSLFIMVC(random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == model.n_clusters
            assert min(labels) == 0
            assert max(labels) == (model.n_clusters - 1)
            assert not np.isnan(labels).any()
            assert model.embedding_.shape == (n_samples, model.n_clusters)
            assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
            assert len(model.beta_) == len(Xs)
            assert model.n_iter_ > 0

def test_custom_parameters(sample_data):
    n_clusters = 3
    model = OSLFIMVC(n_clusters=n_clusters, random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
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
            assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
            assert len(model.beta_) == len(Xs)

def test_invalid_parameters(sample_data):
    with pytest.raises(ValueError, match="Invalid engine."):
        OSLFIMVC(engine='invalid')
    if OCT2PY_INSTALLED:
        with pytest.raises(ValueError, match="Invalid engine."):
            model = OSLFIMVC()
            model.engine = 'invalid'
            model.fit(sample_data[0])

def test_fit_predict(sample_data):
    n_clusters = 3
    model = OSLFIMVC(n_clusters=n_clusters, random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
            assert len(model.beta_) == len(Xs)
            assert model.n_iter_ > 0

def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = OSLFIMVC(n_clusters=n_clusters, random_state=42)
    if OCT2PY_INSTALLED:
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
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
            assert len(model.beta_) == len(Xs)
            assert model.n_iter_ > 0

if __name__ == "__main__":
    pytest.main()