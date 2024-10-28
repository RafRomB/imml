import os
from os.path import dirname
import numpy as np
import pandas as pd
import math
import time
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin

from ..impute import get_observed_view_indicator, simple_view_imputer
from ..utils import check_Xs

oct2py_installed = False
oct2py_module_error = "Oct2Py needs to be installed to use matlab engine."
try:
    import oct2py
    oct2py_installed = True
except ImportError:
    pass


class OPIMC(BaseEstimator, ClassifierMixin):
    r"""
    One-Pass Incomplete Multi-View Clustering (OPIMC).

    OPIMC deals with large scale incomplete multi-view clustering problem by considering the instance missing
    information with the help of regularized matrix factorization and weighted matrix factorization.

    It is recommended to normalize (Normalizer or NormalizerNaN in case incomplete views) the data before applying
    this algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    alpha : float, default=10
        Nonnegative parameter.
    max_iter : int, default=30
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance of the stopping condition.
    block_size : int, default=50
        Size of the chunk.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Current options are 'matlab'.
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        Consensus clustering matrix to be used as input for the KMeans clustering step.

    References
    ----------
    .. [#opimcpaper1] Hu, M., & Chen, S. (2019). One-Pass Incomplete Multi-View Clustering. Proceedings of the AAAI
                     Conference on Artificial Intelligence, 33(01), 3838-3845.
                     https://doi.org/10.1609/aaai.v33i01.33013838.
    .. [#opimcpaper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on
                      Incomplete Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS:
                      SYSTEMS, 2022.
    .. [#opimccode] https://github.com/software-shao/online-multiview-clustering-with-incomplete-view

    Example
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import OPIMC
    >>> from imvc.preprocessing import NormalizerNaN, MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = NormalizerNaN().set_output(transform="pandas")
    >>> estimator = OPIMC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, alpha: float = 10, num_passes: int = 1, max_iter: int = 30,
                 tol: float = 1e-6, block_size: int = 250, random_state:int = None, engine: str ="matlab", verbose = False):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not oct2py_installed):
            raise ImportError(oct2py_module_error)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.num_passes = num_passes
        self.max_iter = max_iter
        self.tol = tol
        self.block_size = block_size
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose

        if self.engine == "matlab":
            matlab_folder = dirname(__file__)
            matlab_folder = os.path.join(matlab_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            matlab_files = [x for x in os.listdir(matlab_folder) if x.endswith(".m")]
            self._oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    self._oc.eval(f.read())


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
            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            observed_view_indicator = get_observed_view_indicator(transformed_Xs)
            transformed_Xs = simple_view_imputer(transformed_Xs, value="zeros")
            transformed_Xs = [X.T for X in transformed_Xs]
            transformed_Xs = tuple(transformed_Xs)

            w = tuple([self._oc.diag(missing_view) for missing_view in observed_view_indicator])
            options = {"block_size": self.block_size, "k": self.n_clusters, "maxiter": self.max_iter,
                       "tol": self.tol, "pass": self.num_passes, "loss": 0, "alpha": self.alpha}
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            labels, V = self._oc.OPIMC(transformed_Xs, w, options, observed_view_indicator, nout= 2)
        elif self.engine=="python":
            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            observed_view_indicator = get_observed_view_indicator(transformed_Xs)
            transformed_Xs = simple_view_imputer(transformed_Xs, value="zeros")
            transformed_Xs = [X.T for X in transformed_Xs]
            transformed_Xs = np.array(tuple(transformed_Xs))

            w = [np.diag(missing_view) for missing_view in observed_view_indicator]
            options = {"block_size": self.block_size, "k": self.n_clusters, "maxiter": self.max_iter,
                       "tol": self.tol, "pass": self.num_passes, "loss": 0, "alpha": self.alpha}
            if self.random_state is not None:
                np.random.seed(self.random_state)
            labels, V = self.opimc(transformed_Xs, w, options, observed_view_indicator)


        self.labels_ = pd.factorize(labels[:,0])[0]
        self.embedding_ = V

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

    def normalize_fea(self, fea, row):
        if not row:
            row = 1

        if row:
            nSmp = fea.shape[0]
            feaNorm = np.maximum(1e-14, np.full(np.sum(fea ** 2, 0)))
            fea = scipy.sparse.spdiags(feaNorm ** -0.5, 0, nSmp, nSmp) * fea
        else:
            nSmp = fea.shape[1]
            feaNorm = np.maximum(1e-14, np.full(np.sum(fea ** 2, 1)))
            fea = fea * scipy.sparse.spdiags(feaNorm ** -0.5, 0, nSmp, nSmp)

        return fea

    def update_v(self, X, W, U, V, viewNum, option):
        n, r = V.shape
        D = np.zeros(shape=(n, r))
        for i in range(viewNum):
            bb = np.full(np.sum(U[i] ** U[i]), 1)
            ab = np.full(np.matmul(X[i].T, U[i]))
            D += W[i] @ (bb[np.ones(shape=(1, n)), :] - 2 * ab)

        _, label = np.nanmin(D, 1)
        # V = np.full(scipy.sparse.csr_matrix([i for i in range(n)], label))

        return V

    def opimc(self, X, W, option, ind):
        num_passes = option["pass"]
        num_views = len(X)
        total = X[0].shape[1]
        block_size = option["block_size"]
        alpha = option["alpha"]

        skip_loss = option["loss"]
        maxIter = option["maxiter"]

        k = option["k"]
        tol = option["tol"]

        index = np.random.permutation(total)
        print(W)
        for i in range(num_views):
            X[i] = X[i][:, np.ix_(index)]
            W[i] = ind[np.ix_(index), i]

        num_feature = np.zeros(shape=(num_views, 1))
        U = []
        for i in range(num_views):
            num_feature[i] = X[i].shape[0]
            U.append(np.random.rand(num_feature[i], k))

        R = []
        T = []

        num_block = math.ceil(total / block_size)
        Loss = np.zeros(num_passes, num_block)

        label_total = np.zeros(num_passes, X[0].shape[1])
        for pass_num in range(num_passes):
            t1 = time.time()
            if pass_num == 1:
                sum_num = 0
                for i in range(num_views):
                    R[i] = np.zeros(shape=(U[i].shape[0], k))
                    T[i] = np.zeros(k)
            else:
                label_total[pass_num, :] = label_total[pass_num - 1, :]

            for block_index in range(num_block):
                data_range = list(range((block_index - 1) * block_size + 1, block_index * block_size + 1))
                V = np.random.rand(block_size, k)
                if block_index == num_block:
                    data_range = list(range((block_index - 1) * block_size + 1, total + 1))
                    V = np.random.rand(total - (num_block - 1) * block_size, k)

                X_block = []
                W_block = []

                for i in range(num_views):
                    X[i][:, data_range] = self.normalize_fea(X[i][:, data_range], 0)
                    X_block.append([X[i][:, data_range]])
                    W_block.append([np.diag(W[i][data_range])])

                if pass_num == 1:
                    sum_num += X[0].shape[1]
                    if block_index == 1:
                        for j in range(num_views):
                            U[j] = np.random.rand(U[j].shape)

                        V = np.random.rand(V.shape)
                        for i in range(V.shape[0]):
                            V[i, :] = V[i, :] / np.sum(V[i, :])

                        _, label_total[pass_num, data_range] = np.maximum(V.T)
                    else:
                        V = self.update_v(X_block, W_block, U, V, num_views, option)
                        _, label_total[pass_num, data_range] = np.maximum(V.T)
                else:
                    V = scipy.sparse.csr_matrix(list(range(0, len(data_range))), label_total[pass_num, data_range])
                    V = V.toarray()

                iter = 0
                converge = 0
                log_out = 0

                for i in range(num_views):
                    tmp1 = R[i] + X_block[i] @ V
                    tmp2 = T[i] + V.T @ W_block[i] @ V
                    log_out -= 2 * np.trace(U[i].T @ tmp1) + np.trace(U[i].T @ U[i] @ tmp2) + \
                               alpha * np.norm(U[i], 'fro') ** 2

                while (iter < maxIter) and converge == 0:
                    if (pass_num != 1) and (iter == 0):
                        V_pre = scipy.sparse.csr_matrix(list(range(0, len(data_range))),
                                                        label_total[pass_num - 1, data_range])

                        for i in range(num_views):
                            T[i] -= V_pre.T @ W_block[i] @ V_pre
                            R[i] -= X_block[i] @ V_pre

                    for i in range(num_views):
                        tmp1 = T[i] + V.T @ W_block[i] @ V + alpha * np.eye(k)
                        tmp2 = R[i] + X_block[i] @ V
                        U_new = tmp2 / tmp1

                        if (pass_num == 1) and (block_index == 1):
                            U_new[:, scipy.sparse.find(np.diag(V.T @ W_block[i] @ V) == 0)] = np.tile(np.mean(
                                X_block[i], 1), reps=len(scipy.sparse.find(np.diag(V.T @ W_block[i] @ V) == 0)))
                            U[i] = U_new
                        else:
                            U_new[:, scipy.sparse.find(np.diag(T[i] @ V.T @ W_block[i] @ V) == 0)] = U[i] \
                                [:, scipy.sparse.find(np.diag(T[i] @ V.T @ W_block[i] @ V) == 0)]
                            U[i] = U_new

                    v = self.update_v(X_block, W_block, U, V, num_views, option)
                    _, label_total[pass_num, data_range] = np.maximum(V.T)
                    log_out_new = 0

                    for i in range(num_views):
                        tmp1 = R[i] + X_block[i] @ V
                        tmp2 = T[i] + V.T @ W_block[i] @ V
                        log_out_new -= 2 * np.trace(U[i].T @ tmp1) + np.trace(U[i].T @ U[i] @ tmp2) + \
                                       alpha * np.norm(U[i], 'fro') ** 2

                    if np.abs((log_out_new - log_out) / log_out) < tol:
                        converge = 1

                    log_out = log_out_new
                    iter += 1

                for i in range(num_views):
                    R[i] += X_block[i] @ V
                    T[i] += V.T @ W_block[i] @ V

                if skip_loss == 0:
                    Loss[pass_num, block_index] = log_out_new / sum_num

        Clu_result = label_total[pass_num, :].T
        return Clu_result, V