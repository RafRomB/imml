import pytest
import numpy as np
import pandas as pd
from imml.preprocessing import SelectCompleteSamples, select_complete_samples


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    indxs = [0, 3]
    Xs_complete_pandas = [X1.iloc[indxs, :], X2.iloc[indxs, :]]
    Xs_complete_numpy = [X1.values[indxs, :], X2.values[indxs, :]]
    return (Xs_pandas, Xs_complete_pandas), (Xs_numpy, Xs_complete_numpy)

def test_select_complete_samples_function(sample_data):
    for Xs, Xs_complete in sample_data:
        transformed_Xs = select_complete_samples(Xs)
        for transformed, expected in zip(transformed_Xs, Xs_complete):
            np.equal(transformed, expected)

def test_select_complete_samples_class(sample_data):
    for Xs, Xs_complete in sample_data:
        transformer = SelectCompleteSamples()
        transformed_Xs = transformer.fit_transform(Xs)
        for transformed, expected in zip(transformed_Xs, Xs_complete):
            np.equal(transformed, expected)

def test_invalid_input():
    with pytest.raises(ValueError):
        select_complete_samples(None)
    with pytest.raises(ValueError):
        select_complete_samples(pd.DataFrame({'feature1': [1, 2]}))
