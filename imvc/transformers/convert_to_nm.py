import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ConvertToNM(BaseEstimator, TransformerMixin):
    r"""
    Convert a matrix into a non-negative matrix as follows:
    1. Create one data matrix with all negative numbers zeroed.
    2. Create another data matrix with all positive numbers zeroed and the signs of all negative numbers removed.
    3. Concatenate both matrices resulting in a data matrix twice as large as the original, but with positive values
    only and zeros and hence appropriate for NMF.

    Attributes
    ----------
    negative_view_ : boolean
        A boolean that indicates whether the input matrix contains any negative values.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset

    >>> from imvc.transformers import ConvertToNM, MultiViewTransformer
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> transformer = MultiViewTransformer(transformer = ConvertToNM())
    >>> transformer.fit_transform(Xs)

    """

    def fit(self, X, y=None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        """

        self.negative_view_ = True if X.lt(0).any().any() else False
        return self


    def transform(self, X):
        r"""
        Transform the input data into a non-negative matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        transformed_X : array-like of shape (n_samples, 2*n_features)
            The transformed data. The shape of the array-like is (n_samples_i, 2*n_features_i) if the view
            has negative values and (n_samples_i, n_features_i) otherwise.
        """

        transformed_X = convert_to_nm(X = X) if self.negative_view_ else X
        return transformed_X


def convert_to_nm(X):
    r"""
    Convert the input matrix data into a positive matrix.

    This function takes a matrix as input and returns a new matrix that has all negative
    values replaced by zeros, and all positive values replaced by zeros and their signs
    removed. The resulting matrix has only positive values and zeros.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    transformed_X : array-like of shape (n_samples, 2*n_features)
    """
    positive_X = X.clip(lower = 0)
    positive_X.columns = positive_X.columns.astype(str) + '_pos'
    negative_X = 0 - X.clip(upper = 0)
    negative_X.columns = negative_X.columns.astype(str) + '_neg'
    transformed_X = pd.concat([positive_X, negative_X], axis=1)
    return transformed_X

