from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class NMFC(NMF, BaseEstimator, ClassifierMixin):
    r"""
    Clustering estimator based on scikit-learn NMC.

    Examples
    --------
    >>> from datasets import load_incomplete_nutrimouse
    >>> from transformers import FillMissingViews
    >>> Xs = load_incomplete_nutrimouse(p = 0)
    >>> estimator = NMFC(n_components = 3)
    >>> estimator.fit_predict(Xs)

    """

    
    def predict(self, X):
        r"""
        Compute clustering by choosing the cluster membership with maximum score in each column of H.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : list of array-likes, shape (n_samples,)
            The predicted data.
        """

        transformed_X = self.transform(X)
        if not isinstance(transformed_X, pd.DataFrame):
            transformed_X = pd.DataFrame(transformed_X)
        transformed_X.columns = np.arange(transformed_X.shape[1])
        labels = transformed_X.idxmax(axis= 1).values
        return labels
    
    
    def fit_predict(self, X, y = None):
        r"""
        Fit the model and predict the clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : list of array-likes, shape (n_samples,)
            The predicted data.
        """
        self.fit(X)
        pred = self.predict(X)
        return pred