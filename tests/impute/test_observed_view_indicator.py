import pytest
import numpy as np
import pandas as pd
from imvc.impute import ObservedViewIndicator, get_observed_view_indicator


@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((5, 3)), index=['a', 'b', 'c', 'd', 'e'],
                      columns=['feature1', 'feature2', 'feature3'])
    X2 = pd.DataFrame(np.random.default_rng(42).random((5, 2)), index=['a', 'b', 'c', 'd', 'e'],
                      columns=['feature4', 'feature5'])
    X1.iloc[[2,4], :] = np.nan
    X2.iloc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    expected_indicator_pandas = pd.DataFrame({
        0: [True, True, False, True, False],
        1: [True, False, True, True, True]
    }, index=['a', 'b', 'c', 'd', 'e'])
    expected_indicator_numpy = expected_indicator_pandas.values
    return Xs_pandas, expected_indicator_pandas, Xs_numpy, expected_indicator_numpy

def test_get_missing_view_indicator(sample_data):
    Xs_pandas, expected_indicator_pandas, Xs_numpy, expected_indicator_numpy = sample_data
    indicator = get_observed_view_indicator(Xs_pandas)
    pd.testing.assert_frame_equal(indicator, expected_indicator_pandas)
    indicator = get_observed_view_indicator(Xs_numpy)
    np.equal(indicator, expected_indicator_numpy)

def test_missing_view_indicator_class(sample_data):
    Xs_pandas, expected_indicator_pandas, Xs_numpy, expected_indicator_numpy = sample_data
    transformer = ObservedViewIndicator()
    indicator = transformer.fit_transform(Xs_pandas)
    pd.testing.assert_frame_equal(indicator, expected_indicator_pandas)
    indicator = transformer.fit_transform(Xs_numpy)
    np.equal(indicator, expected_indicator_numpy)
