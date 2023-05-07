import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import DatasetUtils, check_Xs


class FillMissingViews(BaseEstimator, TransformerMixin):
    r"""
    Fill missing samples in different views of a dataset using a specified method.

    Parameters
    ----------
    value : str, optional (default='mean')
        The method to use for filling missing samples. Possible values:
        - 'mean': replace missing samples with the mean of each feature in the corresponding view
        - 'zeros': replace missing samples with zeros
        - 'nan': replace missing samples with NaN

    Attributes
    ----------
    features_view_mean_list_ : array-like of shape (n_views,)
        The mean value of each feature in the corresponding view, if value='mean'

    Examples
    --------
    >>> from datasets import load_incomplete_nutrimouse
    >>> from transformers import FillMissingViews
    >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
    >>> transformer = FillMissingViews(value = 'mean')
    >>> transformer.fit_transform(Xs)
    """


    def __init__(self, value : str = 'mean'):

        values = ['mean', 'zeros', 'nan']
        if value not in values:
            raise ValueError(f"Invalid value. Expected one of: {values}")
        self.value = value


    def fit(self, Xs, y=None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        """

        Xs = check_Xs(Xs, allow_incomplete=True)
        if self.value == "mean":
            self.features_view_mean_list_ = [X.mean() for X in Xs]
        elif self.value == "zeros":
            pass
        return self


    def transform(self, Xs):
        r"""
        Transform the input data by filling missing samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features)
            The transformed data with filled missing samples.
        """

        Xs = check_Xs(Xs, allow_incomplete=True)
        sample_views = DatasetUtils.get_missing_view_panel(Xs = Xs)
        missing_views = sample_views == 1
        n_samples = len(missing_views)

        transformed_Xs = []
        for X_idx, X in enumerate(Xs):
            n_features = X.shape[1]
            if self.value == "mean":
                feautures_view_mean = self.features_view_mean_list_[X_idx]
                new_X = np.tile(feautures_view_mean, (n_samples ,1))
            elif self.value == "zeros":
                new_X = np.zeros((n_samples, n_features))
            elif self.value == "nan":
                new_X = np.nan
            new_X = pd.DataFrame(new_X, index= missing_views.index, columns = X.columns)
            new_X[missing_views.loc[:, X_idx]] = X
            new_X = new_X.astype(X.dtypes.to_dict())
            transformed_Xs.append(new_X)
        return transformed_Xs
