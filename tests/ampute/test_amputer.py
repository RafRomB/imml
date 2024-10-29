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
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_params(sample_data):
    amputer = Amputer(random_state=42)
    for Xs in sample_data:
        transformed_Xs = amputer.fit_transform(Xs)
        assert DatasetUtils.get_percentage_incomplete_samples(transformed_Xs) == round(amputer.p * 100)
        assert DatasetUtils.get_percentage_complete_samples(transformed_Xs) == round((1 - amputer.p) * 100)
        assert amputer.n_views == len(Xs)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError):
        Amputer(mechanism='Invalid mechanism')

def test_fit(sample_data):
    p = 0.2
    amputer = Amputer(p=p, random_state=42)
    for Xs in sample_data:
        amputer.fit(Xs)
        assert amputer.n_views == len(Xs)

def test_tranform(sample_data):
    p = 0.2
    amputer = Amputer(p=p, random_state=42)
    for Xs in sample_data:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert DatasetUtils.get_percentage_incomplete_samples(transformed_Xs) == round(amputer.p * 100)
        assert DatasetUtils.get_percentage_complete_samples(transformed_Xs) == round((1 - amputer.p) * 100)
        assert amputer.n_views == len(Xs)

def test_extreme_p(sample_data):
    for p in [0, 0.9]:
        amputer = Amputer(p=p, random_state=42)
        for Xs in sample_data:
            Xs = Xs[:2]
            transformed_Xs = amputer.fit_transform(Xs)
            assert DatasetUtils.get_percentage_incomplete_samples(transformed_Xs) == round(amputer.p*100)
            assert DatasetUtils.get_percentage_complete_samples(transformed_Xs) == round((1 - amputer.p)*100)
            assert amputer.n_views == len(Xs)

def test_param_mechanism(sample_data):
    p = 0.2
    for mechanism in ["um", "mcar", "pm", "mnar"]:
        amputer = Amputer(p=p, mechanism=mechanism, random_state=42)
        for Xs in sample_data:
            transformed_Xs = amputer.fit_transform(Xs)
            assert len(transformed_Xs) == len(Xs)
            assert DatasetUtils.get_percentage_incomplete_samples(transformed_Xs) == round(amputer.p*100)
            assert DatasetUtils.get_percentage_complete_samples(transformed_Xs) == round((1 - amputer.p)*100)
            missing = [np.isnan(transformed_X).sum().sum() > 0
                       for transformed_X, X in zip(transformed_Xs, Xs)]
            for transformed_X, X in zip(transformed_Xs, Xs):
                assert transformed_X.shape == X.shape
                if mechanism == "pm":
                    assert sum(missing) == len(Xs) -1
                else:
                    assert sum(missing) == len(Xs)

if __name__ == "__main__":
    pytest.main()