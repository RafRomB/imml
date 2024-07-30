import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import get_observed_view_indicator
from ..utils import check_Xs, DatasetUtils


class IMSCAGL(BaseEstimator, ClassifierMixin):
    r"""
    Incomplete Multiview Spectral Clustering With Adaptive Graph Learning (IMSCAGL).

    IMSCAGL utilizes graph learning and spectral clustering techniques to derive a unified representation for
    incomplete multiview clustering.

    It is recommended to normalize (Normalizer or NormalizerNaN in case incomplete views) the data before applying
    this algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    lambda1 : float, default=0.1
        Penalty parameter for learning model of the multiview subspace clustering.
    lambda2 : float, default=1000
        Penalty parameter for learning model of the multiview subspace clustering.
    lambda3 : float, default=100
        Penalty parameter for learning the consensus representation from those cluster indicator matrices of all views.
    k : int, default=5
        Parameter k of KNN graph.
    neighbor_mode : str, default='KNN'
        Indicates how to construct the graph. Options are 'KNN' (default), and 'Supervised'.
    weight_mode : str, default='Binary'
        Indicates how to assign weights for each edge in the graph. Options are 'Binary' (default), 'Cosine' and 'HeatKernel'.
    max_iter : int, default=100
        Maximum number of iterations.
    miu : float, default=0.01
        Constant for updating variables during the learning process.
    rho : float, default=100
        Constant for updating variables during the learning process.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Currently only 'matlab' is supported.
.   verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : np.array
        Consensus representation matrix to be used as input for the KMeans clustering step.

    References
    ----------
    [paper1] J. Wen, Y. Xu and H. Liu, "Incomplete Multiview Spectral Clustering With Adaptive Graph Learning," in IEEE
            Transactions on Cybernetics, vol. 50, no. 4, pp. 1418-1429, April 2020, doi: 10.1109/TCYB.2018.2884715.
    [paper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on Incomplete
             Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS, 2022.
    [code1]  https://github.com/DarrenZZhang/Survey_IMC
    [code2]  https://github.com/ckghostwj/Incomplete-Multiview-Spectral-Clustering-with-Adaptive-Graph-Learning

    Example
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import IMSCAGL
    >>> from imvc.preprocessing import NormalizerNaN, MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = NormalizerNaN().set_output(transform="pandas")
    >>> estimator = IMSCAGL(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, lambda1: float = 0.1, lambda2: float = 1000, lambda3: float = 100, k: int = 5,
                 neighbor_mode: str = 'KNN', weight_mode: str = 'Binary', max_iter: int = 100, miu: float = 0.01,
                 rho: float = 1.1, random_state: int = None, engine: str = "matlab", verbose = False):
        self.n_clusters = n_clusters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.miu = miu
        self.rho = rho
        self.beta = rho
        self.k = k
        self.neighbor_mode = neighbor_mode
        self.weight_mode = weight_mode
        self.max_iter = max_iter
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose


    def fit(self, Xs, y=None):
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
        self :  Fitted estimator.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')

        if self.engine=="matlab":
            import oct2py
            matlab_folder = dirname(__file__)
            matlab_folder = os.path.join(matlab_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            matlab_files = [x for x in os.listdir(matlab_folder) if x.endswith(".m")]
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            observed_view_indicator = get_observed_view_indicator(Xs=Xs)
            transformed_Xs = DatasetUtils.remove_missing_sample_from_view(Xs=Xs)
            w = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            w = [eye.loc[samples,:].values for eye, (_, samples) in zip(w, observed_view_indicator.items())]
            w = tuple(w)

            transformed_Xs = tuple([X.T for X in transformed_Xs])

            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            F = oc.IMSAGL(transformed_Xs, w, self.n_clusters, self.lambda1, self.lambda2, self.lambda3,
                          self.miu, self.rho, self.max_iter,
                          {"NeighborMode": self.neighbor_mode, "WeightMode": self.weight_mode, "k": self.k})
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.labels_ = model.fit_predict(X= F)
        self.embedding_ = F

        return self


    def _predict(self, Xs):
        r"""
        Return clustering results for samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.labels_


    def fit_predict(self, Xs, y=None):
        r"""
        Fit the model and return clustering results.
        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels
