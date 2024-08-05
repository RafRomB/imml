import pytest
import numpy as np
import pandas as pd
from imvc.preprocessing import SelectCompleteSamples, select_complete_samples


@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((5, 3)), index=['a', 'b', 'c', 'd', 'e'],
                      columns=['feature1', 'feature2', 'feature3'])
    X2 = pd.DataFrame(np.random.default_rng(42).random((5, 2)), index=['a', 'b', 'c', 'd', 'e'],
                      columns=['feature4', 'feature5'])
    X1.iloc[[2,4], :] = np.nan
    X2.iloc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    indxs = [0, 3]
    Xs_complete_pandas = [X1.iloc[indxs, :], X2.iloc[indxs, :]]
    Xs_complete_numpy = [X1.values[indxs, :], X2.values[indxs, :]]
    return Xs_pandas, Xs_complete_pandas, Xs_numpy, Xs_complete_numpy


def test_select_complete_samples_function(sample_data):
    Xs_pandas, Xs_complete_pandas, Xs_numpy, Xs_complete_numpy = sample_data
    transformed_Xs = select_complete_samples(Xs_pandas)
    for transformed, expected in zip(transformed_Xs, Xs_complete_pandas):
        pd.testing.assert_frame_equal(transformed, expected)
    transformed_Xs = select_complete_samples(Xs_complete_numpy)
    for transformed, expected in zip(transformed_Xs, Xs_complete_numpy):
        np.equal(transformed, expected)


def test_select_complete_samples_class(sample_data):
    Xs_pandas, Xs_complete_pandas, Xs_numpy, Xs_complete_numpy = sample_data
    transformer = SelectCompleteSamples()
    transformed_Xs = transformer.fit_transform(Xs_pandas)
    for transformed, expected in zip(transformed_Xs, Xs_complete_pandas):
        pd.testing.assert_frame_equal(transformed, expected)
    transformed_Xs = transformer.fit_transform(Xs_numpy)
    for transformed, expected in zip(transformed_Xs, Xs_complete_numpy):
        np.equal(transformed, expected)

def test_invalid_input():
    with pytest.raises(ValueError):
        select_complete_samples(None)
    with pytest.raises(ValueError):
        select_complete_samples(pd.DataFrame({'feature1': [1, 2]}))
