import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.feature_selection import JNMFFeatureSelector

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    rmodule_installed = True
except ImportError:
    rmodule_installed = False
    rmodule_error = "Module 'r' needs to be installed to use r engine."


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 50))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :20], X.iloc[:, 20:30], X.iloc[:, 30:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_params(sample_data):
    if rmodule_installed:
        transformer = JNMFFeatureSelector(random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'selected_features_')
            assert hasattr(transformer, 'weights_')
            assert (len(transformer.selected_features_) == transformer.n_components)
            assert (len(transformer.selected_features_) == len(transformer.weights_))

def test_fit(sample_data):
    n_components = 5
    if rmodule_installed:
        transformer = JNMFFeatureSelector(n_components=n_components, max_iter=10, random_state=42)
        for Xs in sample_data:
            transformer.fit(Xs)
            assert hasattr(transformer, 'selected_features_')
            assert hasattr(transformer, 'weights_')
            assert (len(transformer.selected_features_) == transformer.n_components)
            assert (len(transformer.selected_features_) == len(transformer.weights_))

def test_transform(sample_data):
    n_components = 5
    if rmodule_installed:
        transformer = JNMFFeatureSelector(n_components=n_components, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            transformer.fit(Xs)
            transformed_X = transformer.transform(Xs)
            transformed_X = np.concatenate(transformed_X, axis=1)
            assert transformed_X.shape == (n_samples, n_components)

def test_param_selectby(sample_data):
    n_components = 5
    if rmodule_installed:
        for select_by in ["max", "component", "average"]:
            transformer = JNMFFeatureSelector(n_components=n_components, select_by=select_by,
                                              random_state=42)
            for Xs in sample_data:
                n_samples = len(Xs[0])
                transformed_X = transformer.fit_transform(Xs)
                transformed_X = np.concatenate(transformed_X, axis=1)
                assert transformed_X.shape == (n_samples, n_components)
                assert hasattr(transformer, 'selected_features_')
                assert hasattr(transformer, 'weights_')
                assert (len(transformer.selected_features_) == transformer.n_components)
                assert (len(transformer.selected_features_) == len(transformer.weights_))

def test_missing_values_handling(sample_data):
    n_components = 5
    if rmodule_installed:
        transformer = JNMFFeatureSelector(n_components=n_components, random_state=42)
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            transformed_X = transformer.fit_transform(Xs)
            transformed_X = np.concatenate(transformed_X, axis=1)
            assert transformed_X.shape == (n_samples, n_components)
            assert hasattr(transformer, 'selected_features_')
            assert hasattr(transformer, 'weights_')
            assert (len(transformer.selected_features_) == transformer.n_components)
            assert (len(transformer.selected_features_) == len(transformer.weights_))

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid select_by"):
        JNMFFeatureSelector(select_by="invalid")


if __name__ == "__main__":
    pytest.main()