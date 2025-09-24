import pytest
import numpy as np
import pandas as pd

from imml.preprocessing import SelectIncompleteSamples, select_incomplete_samples


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


def test_select_incomplete_samples_class(sample_data):
    for Xs in sample_data:
        transformer = SelectIncompleteSamples()
        transformed_Xs = transformer.fit_transform(Xs)
        expected_values = [4, 4, 4]
        for transformed, expected_value in zip(transformed_Xs, expected_values):
            np.equal(transformed, expected_value)


def test_select_incomplete_samples_function(sample_data):
    for Xs in sample_data:
        transformed_Xs = select_incomplete_samples(Xs)
        expected_values = [4, 4, 4]
        for transformed, expected_value in zip(transformed_Xs, expected_values):
            np.equal(transformed, expected_value)


if __name__ == "__main__":
    pytest.main()