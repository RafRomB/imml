import pytest
import numpy as np
import pandas as pd
from imvc.impute import SimpleViewImputer, simple_view_imputer

@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((5, 3)), index=['a', 'b', 'c', 'd', 'e'],
                      columns=['feature1', 'feature2', 'feature3'])
    X2 = pd.DataFrame(np.random.default_rng(42).random((5, 2)), index=['a', 'b', 'c', 'd', 'e'],
                      columns=['feature4', 'feature5'])
    X1.iloc[[2,4], :] = np.nan
    X2.iloc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    indxs = [[2, 4], 1]
    means = [[0.640570, 0.301285, 0.920328], [0.439347, 0.662738]]
    return Xs_pandas, Xs_numpy, indxs, means

def test_mean_imputation(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    imputer = SimpleViewImputer(value='mean')
    transformed_Xs = imputer.fit_transform(Xs_pandas)
    assert len(transformed_Xs) == 2
    assert len(transformed_Xs) == len(imputer.features_view_mean_list_)
    for X_transformed,idx, features_view_mean, mes in zip(transformed_Xs, indxs, imputer.features_view_mean_list_, means):
        assert X_transformed.shape[1] == len(features_view_mean)
        assert not np.isnan(X_transformed).any().any()
        assert np.allclose(features_view_mean, mes)
        assert np.allclose(X_transformed.iloc[idx], mes)

    imputer = SimpleViewImputer(value='mean')
    transformed_Xs = imputer.fit_transform(Xs_numpy)
    assert len(transformed_Xs) == 2
    assert len(transformed_Xs) == len(imputer.features_view_mean_list_)
    for X_transformed,idx, features_view_mean, mes in zip(transformed_Xs, indxs, imputer.features_view_mean_list_, means):
        assert X_transformed.shape[1] == len(features_view_mean)
        assert not np.isnan(X_transformed).any().any()
        assert np.allclose(features_view_mean, mes)
        assert np.allclose(X_transformed[idx], mes)


def test_zeros_imputation(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    imputer = SimpleViewImputer(value='zeros')
    transformed_Xs = imputer.fit_transform(Xs_pandas)
    assert len(transformed_Xs) == 2
    for X_transformed,idx in zip(transformed_Xs, indxs):
        assert not np.isnan(X_transformed).any().any()
        assert (X_transformed.iloc[idx] == 0).all().all()

    imputer = SimpleViewImputer(value='zeros')
    transformed_Xs = imputer.fit_transform(Xs_numpy)
    assert len(transformed_Xs) == 2
    for X_transformed,idx in zip(transformed_Xs, indxs):
        assert not np.isnan(X_transformed).any().any()
        assert (X_transformed[idx] == 0).all().all()

def test_invalid_value():
    with pytest.raises(ValueError, match="Invalid value. Expected one of:"):
        SimpleViewImputer(value='invalid')

def test_transform_without_fit(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    imputer = SimpleViewImputer(value='mean')
    with pytest.raises(AttributeError):
        imputer.transform(Xs_pandas)

def test_mean_imputation_function(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    transformed_Xs = simple_view_imputer(Xs_pandas, value='mean')
    assert len(transformed_Xs) == 2
    for X_transformed,idx, mes in zip(transformed_Xs, indxs, means):
        assert not np.isnan(X_transformed).any().any()
        assert np.allclose(X_transformed.iloc[idx], mes)

    transformed_Xs = simple_view_imputer(Xs_numpy, value='mean')
    assert len(transformed_Xs) == 2
    for X_transformed,idx, mes in zip(transformed_Xs, indxs, means):
        assert not np.isnan(X_transformed).any().any()
        assert np.allclose(X_transformed[idx], mes)

def test_zeros_imputation_function(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    transformed_Xs = simple_view_imputer(Xs_pandas, value='zeros')
    assert len(transformed_Xs) == 2
    for X_transformed,idx in zip(transformed_Xs, indxs):
        assert not np.isnan(X_transformed).any().any()
        assert (X_transformed.iloc[idx] == 0).any().any()

    imputer = SimpleViewImputer(value='zeros')
    transformed_Xs = imputer.fit_transform(Xs_numpy)
    assert len(transformed_Xs) == 2
    for X_transformed,idx in zip(transformed_Xs, indxs):
        assert not np.isnan(X_transformed).any().any()
        assert (X_transformed[idx] == 0).any().any()

def test_invalid_value_function(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    with pytest.raises(ValueError, match="Invalid value. Expected one of:"):
        simple_view_imputer(Xs_pandas, value='invalid')


if __name__ == "__main__":
    pytest.main()