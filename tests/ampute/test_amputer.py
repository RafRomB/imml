import pytest
import numpy as np
import pandas as pd
from string import ascii_lowercase
from imvc.ampute import Amputer
from imvc.impute import get_missing_view_indicator
from imvc.utils import DatasetUtils


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X, index=list(ascii_lowercase)[:len(X)], columns= [f"feature{i}" for i in range(X.shape[1])])
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_initialization():
    amputer = Amputer(p=0.2, mechanism='MCAR', opt='logistic', p_obs=0.1, q=0.3, exclude_inputs=False,
                      p_params=0.4, cut='upper', mcar=True, random_state=42)
    assert amputer.p == 0.2
    assert amputer.mechanism == 'MCAR'
    assert amputer.opt == 'logistic'
    assert amputer.p_obs == 0.1
    assert amputer.q == 0.3
    assert not amputer.exclude_inputs
    assert amputer.p_params == 0.4
    assert amputer.cut == 'upper'
    assert amputer.mcar
    assert amputer.random_state == 42

    with pytest.raises(ValueError):
        Amputer(p=0.2, mechanism='INVALID_MECHANISM')
    with pytest.raises(ValueError):
        Amputer(p=0.2, opt='INVALID_OPT')

def test_fit(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.2, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        assert amputer.n_views == len(Xs)

def test_transform_edm(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.2, mechanism='EDM', random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        missing_view_indicator = get_missing_view_indicator(transformed_Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 16
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 4
        assert pd.Series(missing_view_indicator.sum(axis=0)).between(2,3).all()
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape
            assert np.isnan(transformed_X).sum().sum() > 0

def test_transform_mcar(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.8, mechanism='MCAR', random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 4
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 16
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape

def test_transform_pm(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.8, mechanism='PM', random_state=42)
    for Xs in [Xs_pandas, Xs_numpy, Xs_pandas[:2], Xs_numpy[:2]]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        missing_view_indicator = get_missing_view_indicator(transformed_Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 4
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 16
        assert (missing_view_indicator.sum(axis=0) == 0).sum(axis=0) == 1
        if len(Xs) > 2:
            assert (missing_view_indicator.sum(axis=0) > 0).sum(axis=0) == 2
        else:
            assert (missing_view_indicator.sum(axis=0) > 0).sum(axis=0) == 1
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape

# def test_transform_mar(sample_data):
#     Xs_pandas, Xs_numpy = sample_data
#     amputer = Amputer(p=0.5, mechanism='MAR', opt='logistic', random_state=42)
#     for Xs in [Xs_pandas, Xs_numpy]:
#         amputer.fit(Xs)
#         transformed_Xs = amputer.transform(Xs)
#         assert len(transformed_Xs) == len(Xs)
#         assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 10
#         assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 10
#         for transformed_X, X in zip(transformed_Xs, Xs):
#             assert transformed_X.shape == X.shape
#             assert np.isnan(transformed_X).sum().sum() > 0
#
#     with pytest.raises(ValueError):
#         Amputer(p=0.1, mechanism='MAR').fit_transform(Xs_pandas)


def test_transform_mnar_logistic(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.2, mechanism='MNAR', opt='logistic', random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 16
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 4
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape
            assert np.isnan(transformed_X).sum().sum() > 0

def test_transform_mnar_quantile(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.2, mechanism='MNAR', opt='quantile', q=0.3, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 16
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 4
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape
            assert np.isnan(transformed_X).sum().sum() > 0

    amputer = Amputer(p=0.2, mechanism='MNAR', opt='quantile', q=0.3, random_state=42, cut= "lower", mcar=True)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 16
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 4
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape
            assert np.isnan(transformed_X).sum().sum() > 0

    amputer = Amputer(p=0.2, mechanism='MNAR', opt='quantile', q=0.3, random_state=42, cut= "upper")
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 16
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 4
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape
            assert np.isnan(transformed_X).sum().sum() > 0


def test_transform_mnar_selfmasked(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    amputer = Amputer(p=0.2, mechanism='MNAR', opt='selfmasked', random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert DatasetUtils.get_n_complete_samples(transformed_Xs) == 16
        assert DatasetUtils.get_n_incomplete_samples(transformed_Xs) == 4
        for transformed_X, X in zip(transformed_Xs, Xs):
            assert transformed_X.shape == X.shape

if __name__ == "__main__":
    pytest.main()