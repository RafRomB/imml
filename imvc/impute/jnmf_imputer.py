import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from ..decomposition import jNMF


class jNMFImputer(jNMF):
    r"""
    Impute missing data in multi-view datasets using the Joint Non-negative Matrix Factorization (jNMF) method.

    By decomposing the dataset into joint low-dimensional representations, this method can effectively fill in
    incomplete samples in a way that leverages shared structure across different data views. It supports both
    block-wise and feature-wise missing data imputation.

    References
    ----------
    .. [#jnmfpaper1] Liviu Badea, (2008) Extracting Gene Expression Profiles Common to Colon and Pancreatic
                    Adenocarcinoma using Simultaneous nonnegative matrix factorization. Pacific Symposium on
                    Biocomputing 13:279-290.
    .. [#jnmfpaper2] Shihua Zhang, et al. (2012) Discovery of multi-dimensional modules by integrative analysis of
                     cancer genomic data. Nucleic Acids Research 40(19), 9379-9391.
    .. [#jnmfpaper3] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in
                     heterogeneous omics multi-modal data, Bioinformatics 32(1), 1-8.
    .. [#jnmfpaper4] Y. Kenan Yilmaz et al., (2010) Probabilistic Latent Tensor Factorization, International Conference
                     on Latent Variable Analysis and Signal Separation 346-353.
    .. [#jnmfpaper5] N. Fujita et al., (2018) Biomarker discovery by integrated joint non-negative matrix factorization
                     and pathway signature analyses, Scientific Report.
    .. [#jnmfcode1] https://rdrr.io/cran/nnTensor/man/jNMF.html
    .. [#jnmfcode2] https://github.com/rikenbit/nnTensor

    Example
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.feature_selection import jNMFFeatureSelector
    >>> from imvc.preprocessing import MultiViewTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> transformer = jNMFFeatureSelector(n_components = 5).set_output(transform="pandas")
    >>> pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")), transformer)
    >>> transformed_Xs = pipeline.fit_transform(Xs)
    """


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
        transformed_Xs = [np.dot(transformed_X + V, H.T)
                          for transformed_X,V,H in zip(super().transform(Xs), self.V_, self.H_)]

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        return transformed_Xs


    def fit_transform(self, Xs, y = None, **fit_params):
        r"""
        Fit to data, then transform it.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : Ignored
            Not used, present here for API consistency by convention.
        fit_params : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        transformed_X : array-likes of shape (n_samples, n_components)
            The projected data.
        """

        transformed_Xs_jnmf = [SimpleImputer().set_output(transform="pandas").fit_transform(X) for X in Xs]
        transformed_Xs_jnmf = super().fit_transform(transformed_Xs_jnmf)
        transformed_Xs = []
        for V, H in zip(self.V_, self.H_):
            transformed_X = np.dot(transformed_Xs_jnmf + V, H.T)
            transformed_Xs.append(transformed_X)

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]

        return transformed_Xs

