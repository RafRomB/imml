import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from ..utils import check_Xs


class ObservedViewIndicator(TransformerMixin, BaseEstimator):
    r"""
    Binary indicators for observed views.

    Note that this component typically should not be used in a vanilla Pipeline consisting of transformers and
    an estimator.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.impute import ObservedViewIndicator
    >>> from imvc.ampute import Amputer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> Xs = Amputer(p= 0.2, random_state=42).fit_transform(Xs)
    >>> transformer = ObservedViewIndicator().set_output(transform="pandas")
    >>> X_tr = transformer.fit_transform(Xs)
    """


    def __init__(self):
        self.transform_ = None


    def fit(self, Xs, y = None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self :  returns and instance of self.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        return self


    def transform(self, Xs):
        r"""
        Generate missing view indicator for Xs.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_X : array-like, shape (n_samples, n_views)
            The missing indicator for input data. The data type of transformed_X will be boolean.
        """

        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        transformed_X = np.vstack([np.isnan(X).all(1) for X in Xs]).T
        transformed_X = ~transformed_X
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index= Xs[0].index)
        return transformed_X


    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self
