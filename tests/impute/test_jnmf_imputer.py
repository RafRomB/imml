import pytest
import numpy as np
import pandas as pd
from imvc.impute import jNMFImputer


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((50, 270))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :100].copy(), X.iloc[:, 100:190].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    return Xs_pandas, Xs_numpy

def test_default_params(sample_data):
    transformer = jNMFImputer(random_state=42)
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape

def test_transform(sample_data):
    transformer = jNMFImputer(random_state=42)
    for Xs in sample_data:
        transformer.fit(Xs)
        transformed_Xs = transformer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape

if __name__ == "__main__":
    pytest.main()