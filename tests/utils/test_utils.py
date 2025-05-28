import importlib
import sys
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

from imml.utils import check_Xs, _convert_df_to_r_object

try:
    from rpy2.robjects.packages import importr
    rmodule_installed = True
except ImportError:
    rmodule_installed = False


def test_rmodule_installed():
    df = pd.DataFrame([[1, 2], [3, 4]])
    if rmodule_installed:
        _convert_df_to_r_object(df)
        with patch.dict(sys.modules, {"rpy2": None}):
            import imml.utils.utils as module_mock
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'r' needs to be installed to use r engine."):
                _convert_df_to_r_object(df)
        importlib.reload(module_mock)
    else:
        with pytest.raises(ImportError, match="Module 'r' needs to be installed to use r engine."):
            _convert_df_to_r_object(df)


def test_valid_inputs():
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
    result = check_Xs([df1, df2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    X1 = np.array([[1, 2], [3, 4]])
    result = check_Xs(X1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (2, 2)

    X1 = np.array([[1, 2], [3, 4]])
    X2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
    result = check_Xs([X1, X2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)

    X3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = check_Xs(X3)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    # Test with arrays containing NaN values
    X1 = np.array([[1, np.nan], [3, 4]])
    X2 = np.array([[5, 6], [np.nan, 8]])
    result = check_Xs([X1, X2], force_all_finite='allow-nan')
    assert isinstance(result, list)
    assert len(result) == 2
    assert np.isnan(result[0][0, 1])
    assert np.isnan(result[1][1, 0])


def test_invalid_inputs():
    with pytest.raises(ValueError, match="If not list, input must be of type np.ndarray"):
        check_Xs(123)

    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    with pytest.raises(ValueError, match="Wrong number of modalities. Expected 3 but found 2"):
        check_Xs([X1, X2], enforce_modalities=3)


def test_edge_cases():
    with pytest.raises(ValueError, match="Length of input list must be greater than 0"):
        check_Xs([])


def test_optional_parameters():
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2], copy=True)
    assert not np.may_share_memory(result[0], X1)
    assert not np.may_share_memory(result[1], X2)

    X1 = np.array([[1, np.nan], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2], force_all_finite='allow-nan')
    assert np.isnan(result[0][0, 1])

    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs([X1, X2], return_dimensions=True)
    assert len(result) == 4
    assert result[1] == 2
    assert result[2] == 2
    assert result[3] == [2, 2]


if __name__ == "__main__":
    pytest.main()
