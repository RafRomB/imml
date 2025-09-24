import numpy as np
import pandas as pd
import pytest

from imml.preprocessing import RemoveIncomSamplesByMod, remove_incom_samples_by_mod


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    X3.loc[5, 8:] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy


def test_remove_incom_samples_by_mod_class(sample_data):
    for Xs in sample_data[:2]:
        transformer = RemoveIncomSamplesByMod()
        transformed_Xs = transformer.fit_transform(Xs)
        expected_values = [18, 19, 19]
        for transformed_X, expected_value in zip(transformed_Xs, expected_values):
            assert len(transformed_X) == expected_value


def test_remove_incom_samples_by_mod_function(sample_data):
    for Xs in sample_data[:2]:
        transformed_Xs = remove_incom_samples_by_mod(Xs)
        expected_values = [18, 19, 19]
        for transformed_X, expected_value in zip(transformed_Xs, expected_values):
            assert len(transformed_X) == expected_value


if __name__ == "__main__":
    pytest.main()