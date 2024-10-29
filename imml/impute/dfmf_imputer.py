import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from ..decomposition import jNMF
from ..decomposition._skfusion import fusion
from ..utils import check_Xs


class DFMFImputer(jNMF):
    r"""
    Impute missing data in multi-view datasets using the Joint Non-negative Matrix Factorization (jNMF) method.

    By decomposing the dataset into joint low-dimensional representations, this method can effectively fill in
    incomplete samples in a way that leverages shared structure across different data views. It supports both
    block-wise and feature-wise missing data imputation.

    Example
    --------
    >>> from imml.datasets import LoadDataset
    >>> from imml.feature_selection import jNMFFeatureSelector
    >>> from imml.preprocessing import MultiViewTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> transformer = jNMFFeatureSelector(n_components = 5).set_output(transform="pandas")
    >>> pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")), transformer)
    >>> transformed_Xs = pipeline.fit_transform(Xs)
    """


    def __init__(self, filling: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filling = filling


    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_components)
            The projected data.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]
        if self.filling:
            Xs = [SimpleImputer().set_output(transform="pandas").fit_transform(X) for X in Xs]
        relations = [fusion.Relation(X.values, self.t_, t) for X,t in zip(Xs, self.ts_)]
        transformed_Xs = [self.fuser_.complete(relation) for relation in relations]

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        return transformed_Xs
