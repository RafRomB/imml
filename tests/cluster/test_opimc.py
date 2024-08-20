from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import OPIMC

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
    model = OPIMC(random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert min(labels) == 0
            assert not np.isnan(labels).any()
            assert model.embedding_.shape == (n_samples, model.n_clusters)

def test_custom_parameters(sample_data):
    n_clusters = 3
    model = OPIMC(n_clusters=n_clusters, random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert min(labels) == 0
            assert not np.isnan(labels).any()
            assert model.embedding_.shape == (n_samples, n_clusters)

def test_invalid_parameters(sample_data):
    with pytest.raises(ValueError, match="Invalid engine."):
        OPIMC(engine='invalid')
    if OCT2PY_INSTALLED:
        with pytest.raises(ValueError, match="Invalid engine."):
            model = OPIMC()
            model.engine = 'invalid'
            model.fit(sample_data[0])

def test_fit_predict(sample_data):
    n_clusters = 3
    model = OPIMC(n_clusters=n_clusters, random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert min(labels) == 0
            assert not np.isnan(labels).any()
            assert model.embedding_.shape == (n_samples, n_clusters)

def test_missing_values_handling(sample_data):
    n_clusters = 2
    model = OPIMC(n_clusters=n_clusters, random_state=42)
    if OCT2PY_INSTALLED:
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert min(labels) == 0
            assert not np.isnan(labels).any()
            assert model.embedding_.shape == (n_samples, n_clusters)

if __name__ == "__main__":
    pytest.main()