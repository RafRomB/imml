import pytest
import numpy as np
import pandas as pd

from imvc.utils import check_Xs


def test_valid_inputs():
    # Test valid list of numpy arrays
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    # Test valid list of pandas DataFrames
    df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
    result = check_Xs([df1, df2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    # Test 2D numpy array converted to list with single element
    X1 = np.array([[1, 2], [3, 4]])
    result = check_Xs(X1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (2, 2)

    # Test valid list with mixed types (numpy arrays and pandas DataFrames)
    X1 = np.array([[1, 2], [3, 4]])
    X2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
    result = check_Xs([X1, X2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)

    # Test list with 3D numpy array to trigger the conversion path
    X3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = check_Xs(X3)
    assert isinstance(result, list)
    assert len(result) == 2  # The 3D array is split into two 2D arrays
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)


def test_invalid_inputs():
    # Test non-list input
    with pytest.raises(ValueError, match="If not list, input must be of type np.ndarray"):
        check_Xs(123)

    # Test enforcing a specific number of views
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    with pytest.raises(ValueError, match="Wrong number of views. Expected 3 but found 2"):
        check_Xs([X1, X2], enforce_views=3)


def test_edge_cases():
    # Test empty list
    with pytest.raises(ValueError, match="Length of input list must be greater than 0"):
        check_Xs([])


def test_optional_parameters():
    # Test copy parameter
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2], copy=True)
    assert not np.may_share_memory(result[0], X1)
    assert not np.may_share_memory(result[1], X2)

    # Test force_all_finite parameter
    X1 = np.array([[1, np.nan], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2], force_all_finite='allow-nan')
    assert np.isnan(result[0][0, 1])  # Ensure NaN is allowed

    # Test return_dimensions parameter
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2], return_dimensions=True)
    assert len(result) == 4
    assert result[1] == 2  # n_views
    assert result[2] == 2  # n_samples
    assert result[3] == [2, 2]  # n_features


if __name__ == "__main__":
    pytest.main()