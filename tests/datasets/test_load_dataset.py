import pytest
import pandas as pd
from imvc.datasets import LoadDataset


# def test_load_dataset():
#     Xs = LoadDataset.load_dataset('bbcsport')
#     assert isinstance(Xs, list)
#     assert len(Xs) == 4  # 4 views
#
#     Xs, y = LoadDataset.load_dataset('bbcsport', return_y=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 4
#     assert isinstance(y, pd.Series)
#
#     Xs, y, metadata = LoadDataset.load_dataset('bbcsport', return_y=True, return_metadata=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 4
#     assert isinstance(y, pd.Series)
#     assert isinstance(metadata, dict)
#
#
# def test_load_bbcsport():
#     Xs = LoadDataset.load_bbcsport()
#     assert isinstance(Xs, list)
#     assert len(Xs) == 4
#
#     Xs, y = LoadDataset.load_bbcsport(return_y=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 4
#     assert isinstance(y, pd.Series)
#
#     Xs, y, metadata = LoadDataset.load_bbcsport(return_y=True, return_metadata=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 4
#     assert isinstance(y, pd.Series)
#     assert isinstance(metadata, dict)
#
#
# def test_load_bdgp():
#     Xs = LoadDataset.load_bdgp()
#     assert isinstance(Xs, list)
#     assert len(Xs) == 2
#
#     Xs, y = LoadDataset.load_bdgp(return_y=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 2
#     assert isinstance(y, pd.Series)
#
#     Xs, y, metadata = LoadDataset.load_bdgp(return_y=True, return_metadata=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 2
#     assert isinstance(y, pd.Series)
#     assert metadata is None
#
#
# def test_load_buaa():
#     Xs = LoadDataset.load_buaa()
#     assert isinstance(Xs, list)
#     assert len(Xs) == 2
#
#     Xs, y = LoadDataset.load_buaa(return_y=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 2
#     assert isinstance(y, pd.Series)
#
#     Xs, y, metadata = LoadDataset.load_buaa(return_y=True, return_metadata=True)
#     assert isinstance(Xs, list)
#     assert len(Xs) == 2
#     assert isinstance(y, pd.Series)
#     assert metadata is None
#
#
def test_load_nonexistent_dataset():
    with pytest.raises(FileNotFoundError):
        LoadDataset.load_dataset('nonexistent_dataset')


if __name__ == "__main__":
    pytest.main()