import pytest
import numpy as np
import pandas as pd
from imvc.impute.mofa_imputer import MOFAImputer


@pytest.fixture
def sample_data():
    X1 = pd.DataFrame(np.random.default_rng(42).random((50, 100)))
    X2 = pd.DataFrame(np.random.default_rng(42).random((50, 90)))
    X1.iloc[[2,4], :] = np.nan
    X2.iloc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    return Xs_pandas, Xs_numpy

def test_MOFAImputer_default(sample_data):
    transformer = MOFAImputer(random_state=42)
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape

def test_MOFAImputer_params(sample_data):
    transformer = MOFAImputer(n_components=5, random_state=42)
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape

def test_MOFAImputer_set_output(sample_data):
    transformer = MOFAImputer(n_components=5, random_state=42, verbose=True).set_output(transform="pandas")
    assert transformer.transform_ == "pandas"
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape
        assert isinstance(transformed_Xs[1], pd.DataFrame)

if __name__ == "__main__":
    pytest.main()