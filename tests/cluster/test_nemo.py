from unittest import mock
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.cluster import NEMO

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    rpy2_installed = True
except ImportError:
    rpy2_installed = False

if rpy2_installed:
    rbase = importr("base")
    try:
        snftool_installed = True
        snftool = importr("SNFtool")
    except PackageNotInstalledError:
        snftool_installed = False
estimator = NEMO


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((50, 20))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :10], X.iloc[:, 10:15], X.iloc[:, 15:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_rpy2_not_installed(monkeypatch):
    if rpy2_installed:
        estimator(engine="r")
        with mock.patch("imvc.cluster.nemo.rpy2_installed", False):
            with mock.patch("imvc.cluster.nemo.rpy2_module_error",
                            "rpy2 needs to be installed to use r engine."):
                with pytest.raises(ImportError, match="rpy2 needs to be installed to use r engine."):
                    estimator(engine="r")
    else:
        with pytest.raises(ImportError, match="rpy2 needs to be installed to use r engine."):
            estimator(engine="r")

@pytest.mark.skipif(not rpy2_installed, reason="rpy2 is not installed.")
def test_r_dependencies_not_installed():
    if rpy2_installed:
        if snftool_installed:
            NEMO(engine="r")
        else:
            with pytest.raises(ImportError, match="SNFtool needs to be installed in R to use r engine."):
                NEMO(engine="r")

def test_param_randomstate(sample_data):
    random_state = 42
    for engine in ["r", "python"]:
        if (engine == "r") and not rpy2_installed:
            continue
        else:
            labels = estimator(engine=engine, random_state=random_state).fit_predict(sample_data[0])
            assert all(labels == estimator(engine=engine, random_state=random_state).fit_predict(sample_data[0]))

def test_default_params(sample_data):
    model = estimator(random_state=42)
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
    for engine in ["r", "python"]:
        if (engine == "r") and not rpy2_installed:
            continue
        else:
            model = estimator(n_clusters=list(range(2, 4)), num_neighbors=3, random_state=42, engine=engine)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == model.n_clusters_
            assert min(labels) == 0
            assert max(labels) == (model.n_clusters_ - 1)
            assert not np.isnan(labels).any()
            if engine == "python":
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
    for engine in ["r", "python"]:
        if (engine == "r") and not rpy2_installed:
            continue
        else:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(np.unique(labels)) == model.n_clusters_
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            if engine == "python":
                assert not np.isnan(model.embedding_).any().any()
                assert model.embedding_.shape == (n_samples, model.n_clusters_)
            assert model.affinity_matrix_.shape == (n_samples, n_samples)
            assert len(model.num_neighbors_) == len(Xs)
            assert model.num_neighbors_[0] > 0

def test_missing_values_handling(sample_data):
    n_clusters = 3
    for engine in ["r", "python"]:
        if (engine == "r") and not rpy2_installed:
            continue
        else:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(np.unique(labels)) == model.n_clusters_
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            if engine == "python":
                assert not np.isnan(model.embedding_).any().any()
                assert model.embedding_.shape == (n_samples, model.n_clusters_)
            assert model.affinity_matrix_.shape == (n_samples, n_samples)
            assert len(model.num_neighbors_) == len(Xs)
            assert model.num_neighbors_[0] > 0

if __name__ == "__main__":
    pytest.main()