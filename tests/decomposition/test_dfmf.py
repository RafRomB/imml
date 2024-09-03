from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imvc.ampute import Amputer
from imvc.decomposition import DFMF


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X, index=list(ascii_lowercase)[:len(X)], columns= [f"feature{i}" for i in range(X.shape[1])])
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy


def test_dfmf_init(sample_data):
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=200, init_type='random', n_run=3, random_state=42)
    assert dfmf.n_components == n_components
    assert dfmf.max_iter == 200
    assert dfmf.init_type == ['random', 'random']
    assert dfmf.n_run == 3
    assert dfmf.random_state == 42

def test_dfmf_fit(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=10, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        dfmf.fit(Xs)
        assert hasattr(dfmf, 'fuser_')
        assert hasattr(dfmf, 'transformer_')
        assert hasattr(dfmf, 'ts_')
        assert len(dfmf.ts_) == len(Xs)

def test_dfmf_transform(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=10, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        dfmf.fit(Xs)
        transformed_X = dfmf.transform(Xs)
        assert transformed_X.shape == (n_samples, n_components)

def test_dfmf_set_output(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    dfmf = DFMF(n_components=5, max_iter=10, random_state=42).set_output(transform="pandas")
    assert dfmf.transform_ == "pandas"
    for Xs in [Xs_pandas, Xs_numpy]:
        transformed_X = dfmf.fit_transform(Xs)
        assert isinstance(transformed_X, pd.DataFrame)

def test_missing_values_handling(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=10, random_state=42).set_output(transform="pandas")
    for Xs in [Xs_pandas, Xs_numpy]:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        transformed_X = dfmf.fit_transform(Xs)
        assert not np.isnan(transformed_X).any().any()
        assert transformed_X.shape == (n_samples, n_components)

def test_dfmf_invalid_params():
    with pytest.raises(ValueError, match="Invalid n_components"):
        DFMF(n_components=0)


if __name__ == "__main__":
    pytest.main()