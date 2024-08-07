import pytest
import numpy as np
import pandas as pd
from imvc.utils import DatasetUtils

@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((5, 3)), index=['a', 'b', 'c', 'd', 'e'], columns=['feature1', 'feature2', 'feature3'])
    X2 = pd.DataFrame(np.random.default_rng(42).random((5, 2)), index=['a', 'b', 'c', 'd', 'e'], columns=['feature4', 'feature5'])
    observed_view_indicator_pandas = pd.DataFrame({
        0: [True, True, False, True, False],
        1: [True, False, True, True, True]
    }, index=['a', 'b', 'c', 'd', 'e'])
    X1.iloc[[2,4], :] = np.nan
    X2.iloc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    observed_view_indicator_numpy = observed_view_indicator_pandas.values
    return Xs_pandas, observed_view_indicator_pandas, Xs_numpy, observed_view_indicator_numpy

def test_convert_to_imvd(sample_data):
    Xs_pandas, observed_view_indicator_pandas, Xs_numpy, observed_view_indicator_numpy = sample_data
    for Xs, observed_view_indicator in zip([Xs_pandas, Xs_numpy], [observed_view_indicator_pandas, observed_view_indicator_numpy]):
        transformed_Xs = DatasetUtils.convert_to_imvd(Xs, observed_view_indicator)
        assert len(transformed_Xs) == len(Xs)
        values_to_compare = [2,1]
        for i, X in enumerate(Xs):
            assert transformed_Xs[i].shape == X.shape
            assert np.isnan(transformed_Xs[i]).all(1).sum() == values_to_compare[i]


def test_get_n_views(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        n_views = DatasetUtils.get_n_views(Xs)
        assert n_views == len(Xs)

def test_get_n_samples_by_view(sample_data):
    Xs_pandas, observed_view_indicator_pandas, Xs_numpy, observed_view_indicator_numpy = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples_by_view = DatasetUtils.get_n_samples_by_view(Xs)
        values_to_compare = [3,4]
        assert all(n_samples_by_view == values_to_compare)

def test_get_complete_sample_names(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        complete_samples = DatasetUtils.get_complete_sample_names(Xs)
        assert len(complete_samples) == 2
        if isinstance(Xs[0], pd.DataFrame):
            values_to_compare = pd.Index(["a","d"])
            assert complete_samples.equals(values_to_compare)
        else:
            values_to_compare = pd.Index([0,3])
            assert complete_samples.equals(values_to_compare)

def test_get_incomplete_sample_names(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        incomplete_samples = DatasetUtils.get_incomplete_sample_names(Xs)
        assert len(incomplete_samples) == 3
        if isinstance(Xs[0], pd.DataFrame):
            values_to_compare = pd.Index(["b","c","e"])
            assert incomplete_samples.equals(values_to_compare)
        else:
            values_to_compare = pd.Index([1,2,4])
            assert incomplete_samples.equals(values_to_compare)

def test_get_sample_names(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        sample_names = DatasetUtils.get_sample_names(Xs)
        assert len(sample_names) == 5
        if isinstance(Xs[0], pd.DataFrame):
            values_to_compare = pd.Index(["a","b","c","d","e"])
            assert sample_names.equals(values_to_compare)
        else:
            values_to_compare = pd.Index(range(len(sample_names)))
            assert sample_names.equals(values_to_compare)

def test_get_samples_by_view(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        samples_by_view = DatasetUtils.get_samples_by_view(Xs, return_as_list=True)
        assert len(samples_by_view) == len(Xs)
        if isinstance(Xs[0], pd.DataFrame):
            values_to_compare = [["a","b","d"], ["a","c","d", "e"]]
        else:
            values_to_compare = [[0, 1, 3], [0, 2, 3, 4]]
        for i in range(len(samples_by_view)):
            assert samples_by_view[i].equals(pd.Index(values_to_compare[i]))
        samples_by_view = DatasetUtils.get_samples_by_view(Xs, return_as_list=False)
        assert len(samples_by_view) == len(Xs_pandas)
        for i in range(len(samples_by_view)):
            assert samples_by_view[i].equals(pd.Index(values_to_compare[i]))

def test_get_missing_samples_by_view(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        missing_samples_by_view = DatasetUtils.get_missing_samples_by_view(Xs, return_as_list=True)
        assert len(missing_samples_by_view) == len(Xs)
        if isinstance(Xs[0], pd.DataFrame):
            values_to_compare = [["c", "e"], ["b"]]
        else:
            values_to_compare = [[2, 4], [1]]
        for i in range(len(missing_samples_by_view)):
            assert pd.Index(missing_samples_by_view[i]).equals(pd.Index(values_to_compare[i]))
        missing_samples_by_view = DatasetUtils.get_missing_samples_by_view(Xs, return_as_list=False)
        assert len(missing_samples_by_view) == len(Xs)
        for i in range(len(missing_samples_by_view)):
            assert pd.Index(missing_samples_by_view[i]).equals(pd.Index(values_to_compare[i]))

def test_get_n_complete_samples(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        n_complete_samples = DatasetUtils.get_n_complete_samples(Xs)
        assert n_complete_samples == 2

def test_get_n_incomplete_samples(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        n_incomplete_samples = DatasetUtils.get_n_incomplete_samples(Xs)
        assert n_incomplete_samples == 3

def test_get_percentage_complete_samples(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        percentage_complete = DatasetUtils.get_percentage_complete_samples(Xs)
        assert percentage_complete == (2 / 5) * 100

def test_get_percentage_incomplete_samples(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        percentage_incomplete = DatasetUtils.get_percentage_incomplete_samples(Xs)
        assert percentage_incomplete == (3 / 5) * 100

def test_remove_missing_sample_from_view(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        cleaned_Xs = DatasetUtils.remove_missing_sample_from_view(Xs)
        values_to_compare = [3, 4]
        for to_compare, X in zip(values_to_compare, cleaned_Xs):
            assert len(X) == to_compare

def test_convert_mvd_from_list_to_dict(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        Xs_dict = DatasetUtils.convert_mvd_from_list_to_dict(Xs)
        assert isinstance(Xs_dict, dict)
        assert len(Xs_dict) == len(Xs)
        assert all(isinstance(k, int) for k in Xs_dict.keys())

def test_convert_mvd_from_dict_to_list(sample_data):
    Xs_pandas, _, Xs_numpy, _ = sample_data
    for Xs in [Xs_pandas, Xs_numpy]:
        Xs_dict = DatasetUtils.convert_mvd_from_list_to_dict(Xs)
        Xs_list = DatasetUtils.convert_mvd_from_dict_to_list(Xs_dict)
        assert len(Xs_list) == len(Xs)
        for X, X_converted in zip(Xs, Xs_list):
            assert np.array_equal(X, X_converted, equal_nan=True)

if __name__ == "__main__":
    pytest.main()