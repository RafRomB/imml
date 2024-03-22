import os
import oct2py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..transformers import FillIncompleteSamples
from ..utils import check_Xs, DatasetUtils


class DAIMC(BaseEstimator, ClassifierMixin):
    r"""
    Doubly Aligned Incomplete Multi-view Clustering (DAIMC).

    The DAIMC algorithm integrates weighted semi-nonnegative matrix factorization (semi-NMF) to address incomplete
    multi-view clustering challenges. It leverages instance alignment information to learn a unified latent feature
    matrix across views and employs L2,1-Norm regularized regression to establish a consensus basis matrix, minimizing
    the impact of missing instances.

    The recommended preprocessing is applying Normalizer and replacing missing views with 0.

    Parameters
    ----------
    n_clusters : int or list-of-int
        The number of clusters to generate. If it is a list, the number of clusters will be estimated by the algorithm
         with this range of number of clusters to choose between.
    afa : float, default 1e1
        nonnegative.
    beta : float, default 1e0
        Define the trade-off between sparsity and accuracy of regression for the i-th view.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model.
.   verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    u_ : np.array
        Basis matrix.
    v_ : np.array
        Commont latent feature matrix.
    b_ : np.array
        Regression coefficient matrices.

    References
    ----------
    [paper1] Menglei Hu and Songcan Chen. 2018. Doubly aligned incomplete multi-view clustering. In Proceedings of the
            27th International Joint Conference on Artificial Intelligence (IJCAI'18). AAAI Press, 2262–2268.
    [paper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on Incomplete
             Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS, 2022.
    [code]  https://github.com/DarrenZZhang/Survey_IMC

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.preprocessing import Normalizer, FunctionTransformer
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import DAIMC
    >>> from imvc.transformers import MultiViewTransformer

    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = lambda x: x.divide(x.pow(2).sum(axis=1).pow(1/2), axis= 0)
    >>> estimator = DAIMC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(FunctionTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, afa: float = 1e1, beta: float = 1e0, random_state:int = None,
                 engine: str ="matlab", verbose = False):
        self.n_clusters = n_clusters
        self.afa = afa
        self.beta = beta
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
            oc = oct2py.Oct2Py(temp_dir="imvc/cluster/_daimc/")
            with open(os.path.join("imvc", "cluster", "_daimc", "newinit.m")) as f:
                oc.eval(f.read())
            with open(os.path.join("imvc", "cluster", "_daimc", "litekmeans.m")) as f:
                oc.eval(f.read())
            with open(os.path.join("imvc", "cluster", "_daimc", "DAIMC.m")) as f:
                oc.eval(f.read())
            with open(os.path.join("imvc", "cluster", "_daimc", "UpdateV_DAIMC.m")) as f:
                oc.eval(f.read())
            oc.eval("pkg load statistics")
            oc.eval("pkg load control")
            oc.warning("off", "Octave:possible-matlab-short-circuit-operator")

            missing_view_profile = DatasetUtils.get_missing_view_profile(Xs=Xs)
            transformed_train_Xs = FillIncompleteSamples(value="zeros").fit_transform(Xs)
            transformed_train_Xs = [X.T for X in transformed_train_Xs]
            transformed_train_Xs = tuple(transformed_train_Xs)

            w = tuple([oc.diag(missing_view) for _, missing_view in missing_view_profile.items()])
            u_0, v_0, b_0 = oc.newinit(transformed_train_Xs, w, self.n_clusters, len(transformed_train_Xs), nout=3)
            u, v, b, f, p, n = oc.DAIMC(transformed_train_Xs, w, u_0, v_0, b_0, None, self.n_clusters,
                                        len(transformed_train_Xs), {"afa": self.afa, "beta": self.beta}, nout=6)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.labels_ = model.fit_predict(X= v)
        self.u_ = u
        self.v_ = v
        self.b_ = b

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

