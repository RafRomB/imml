import importlib
import sys
from unittest.mock import patch
import pytest
import pandas as pd

from imml.utils import _convert_df_to_r_object

try:
    from rpy2.robjects.packages import importr
    rmodule_installed = True
except ImportError:
    rmodule_installed = False


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Test never ends on Windows")
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


if __name__ == "__main__":
    pytest.main()
