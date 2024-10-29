import numpy as np
import pytest
import pandas as pd
from imml.preprocessing import (
    DropView, ConcatenateViews, SingleView, AddMissingViews, SortData,
    concatenate_views, drop_view, single_view, add_missing_views, sort_data
)
from imml.utils import DatasetUtils

def create_sample_data():
    # Utility function to create sample data
    X1 = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['f1', 'f2'])
    X2 = pd.DataFrame([[5, 6], [7, 8]], index=['a', 'b'], columns=['f3', 'f4'])
    X3 = pd.DataFrame([[9, 10], [11, 12]], index=['a', 'b'], columns=['f5', 'f6'])
    samples = pd.Index(['a', 'b'])
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy, samples

def test_drop_view_transformer():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    transformer = DropView(X_idx=1)
    result = transformer.fit_transform(Xs_pandas)
    assert len(result) == 2
    for X,X_result in zip(Xs_pandas, result):
        assert np.equal(X.shape, X_result.shape).all()
    result = transformer.fit_transform(Xs_numpy)
    assert len(result) == 2
    for X,X_result in zip(Xs_numpy, result):
        assert np.equal(X.shape, X_result.shape).all()

def test_concatenate_views_transformer():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    transformer = ConcatenateViews()
    result = transformer.fit_transform(Xs_pandas)
    assert np.equal(result.shape, (2, 6)).all()
    result = transformer.fit_transform(Xs_numpy)
    assert np.equal(result.shape, (2, 6)).all()

def test_single_view_transformer():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    i = 0
    transformer = SingleView(X_idx=i)
    result = transformer.fit_transform(Xs_pandas)
    assert result.equals(Xs_pandas[i])
    result = transformer.fit_transform(Xs_numpy)
    assert np.equal(result, Xs_numpy[i]).all()

def test_add_missing_views_transformer():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    samples = pd.Index(['a', 'b', 'c'])
    transformer = AddMissingViews(samples=samples)
    result = transformer.fit_transform(Xs_pandas[0])
    assert result.shape == (3, 2)
    assert result.index.equals(pd.Index(['a', 'b', 'c']))
    result = transformer.fit_transform(Xs_numpy[0])
    assert result.shape == (3, 2)

def test_sort_data_transformer(monkeypatch):
    Xs_pandas, Xs_numpy, samples = create_sample_data()
    transformer = SortData()
    result = transformer.fit_transform(Xs_pandas)
    for X_result in result:
        assert all(X_result.index == samples)
    result = transformer.fit_transform(Xs_numpy)
    for X_result in result:
        assert all(X_result.index == list(range(len(samples))))

def test_concatenate_views_function():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    result = concatenate_views(Xs_pandas)
    assert np.equal(result.shape, (2, 6)).all()
    result = concatenate_views(Xs_numpy)
    assert np.equal(result.shape, (2, 6)).all()

def test_drop_view_function():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    result = drop_view(Xs_pandas)
    assert len(result) == 2
    for X,X_result in zip(Xs_pandas, result):
        assert np.equal(X.shape, X_result.shape).all()
    result = drop_view(Xs_numpy)
    assert len(result) == 2
    for X,X_result in zip(Xs_numpy, result):
        assert np.equal(X.shape, X_result.shape).all()

def test_single_view_function():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    i = 0
    result = single_view(Xs_pandas, X_idx=i)
    assert result.equals(Xs_pandas[i])
    result = single_view(Xs_numpy, X_idx=i)
    assert np.equal(result, Xs_numpy[i]).all()

def test_add_missing_views_function():
    Xs_pandas, Xs_numpy, _ = create_sample_data()
    samples = pd.Index(['a', 'b', 'c'])
    result = add_missing_views(Xs_pandas[0], samples=samples)
    assert result.shape == (3, 2)
    assert result.index.equals(pd.Index(['a', 'b', 'c']))
    result = add_missing_views(Xs_numpy[0], samples=samples)
    assert result.shape == (3, 2)

def test_sort_data_function(monkeypatch):
    Xs_pandas, Xs_numpy, samples = create_sample_data()
    result = sort_data(Xs_pandas)
    for X_result in result:
        assert all(X_result.index == samples)
    result = sort_data(Xs_numpy)
    for X_result in result:
        assert all(X_result.index == list(range(len(samples))))

def test_invalid_drop_view_index():
    Xs_pandas, _, _ = create_sample_data()
    with pytest.raises(ValueError, match="X_idx out of range. Should be between 0 and n_views - 1"):
        drop_view(Xs_pandas, X_idx=10)

def test_invalid_single_view_index():
    Xs_pandas, _, _ = create_sample_data()
    with pytest.raises(ValueError, match="X_idx out of range. Should be between 0 and n_views - 1"):
        single_view(Xs_pandas, X_idx=10)

if __name__ == "__main__":
    pytest.main()