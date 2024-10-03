import math
import os
from os.path import dirname
import numpy as np
import pandas as pd

from scipy.sparse.linalg import eigs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.neighbors import NearestNeighbors

from ..utils import check_Xs, DatasetUtils

try:
    import oct2py
    oct2py_installed = True
except ImportError:
    oct2py_installed = False
    oct2py_module_error = "Oct2Py needs to be installed to use matlab engine."


class MKKMIK(BaseEstimator, ClassifierMixin):
    r"""
    Multiple Kernel k-Means with Incomplete Kernels (MKKM-IK).

    MKKM-IK integrates imputation and clustering into a single optimization procedure. Thus, the clustering result
    guides the missing kernel imputation, and the latter is used to conduct the subsequent clustering. Both procedures
    will be performed until convergence.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    kernel : callable, default=kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())
        Specifies the kernel type to be used in the algorithm.
    kernel_initialization : str, default="zeros"
        Specifies the algorithm to initialize the kernel. It should be one of ['zeros', 'mean', 'knn', 'em', 'laplacian'].
    lambda_reg : float, default=1.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    qnorm : float, default=2.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
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
    gamma_ : array-like of shape (n_views,)
        Kernel weights.
    KA_ : array-like of shape (n_samples, n_views)
        Kernel sub-matrix.
    loss_ : array-like of shape (n_iter_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#mkkmikpaper] X. Liu et al., "Multiple Kernel k-Means with Incomplete Kernels," in IEEE Transactions on Pattern
                      Analysis and Machine Intelligence, vol. 42, no. 5, pp. 1191-1204, 1 May 2020,
                      doi: 10.1109/TPAMI.2019.2892416.
    .. [#mkkmikcode] https://github.com/wangsiwei2010/multiple_kernel_clustering_with_absent_kernel

    Example
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import MKKMIK
    >>> from sklearn.preprocessing import StandardScaler
    >>> from imvc.preprocessing import MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = StandardScaler().set_output(transform="pandas")
    >>> estimator = MKKMIK(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)

    """

    def __init__(self, n_clusters: int = 8, kernel_initialization: str = "zeros",
                 kernel: callable = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel()),
                 qnorm: float = 2., random_state: int = None, engine: str = "matlab", verbose=False):

        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not oct2py_installed):
            raise ModuleNotFoundError(oct2py_module_error)
        kernel_initializations = ['zeros', 'mean', 'knn', 'em', 'laplacian']
        if kernel_initialization not in kernel_initializations:
            raise ValueError(f"Invalid kernel_initialization. Expected one of: {kernel_initializations}")

        self.n_clusters = n_clusters
        self.kernel_initialization = kernel_initialization
        self.qnorm = qnorm
        self.kernel = kernel
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose
        self.kernel_initializations = {"zeros": "algorithm2", "mean": "algorithm3", "knn": "algorithm0",
                                       "em": "algorithm6", "laplacian": "algorithm4"}

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

        if self.engine == "matlab":
            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            s = DatasetUtils.get_missing_samples_by_view(Xs=transformed_Xs, return_as_list=True)
            s = tuple([{"indx": pd.Series(i).add(1).to_list()} for i in s])

            transformed_Xs = [self.kernel(X) for X in transformed_Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            kernel = self.kernel_initializations[self.kernel_initialization]

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            H_normalized,gamma,obj,KA = self._oc.myabsentmultikernelclustering(transformed_Xs, s, self.n_clusters,
                                                                         self.qnorm, kernel, nout=4)
            KA = KA[:, 0]
            obj = obj[0]

        model = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=self.random_state)
        self.labels_ = model.fit_predict(X=H_normalized)
        self.embedding_, self.gamma_, self.KA_, self.loss_ = H_normalized, gamma, KA, obj
        self.n_iter_ = len(self.loss_)

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

    def algorithm0(self, K, S, k):
        KM = self.algorithm2(K, S)
        ImputedKM = KM
        numker = KM.shape[2]
        for p in range(numker):
            Indx = np.setdiff1d(ar1=[i for i in range(numker)], ar2=p)
            MissingIndex = [i - 1 for i in S[p]['indx']]
            Tempk = np.sum(KM[:, :, Indx], 2) / numker

            knn = NearestNeighbors(n_neighbors=k + 1)
            knn.fit(Tempk)
            IDX = knn.kneighbors(Tempk[np.ix_(MissingIndex), :], return_distance=False)

            for j in range(len(MissingIndex)):
                heheTemp = np.mean(Tempk[np.ix_(IDX[j, 1:k + 1]), :], axis=0)
                ImputedKM[np.ix_(MissingIndex[j]), :, p] = heheTemp
                ImputedKM[:, np.ix_(MissingIndex[j]), p] = heheTemp

            # Ensure matrix is symmetric
            ImputedKM[:, :, p] = (ImputedKM[:, :, p] + ImputedKM[:, :, p].T) / 2

        return ImputedKM


    def algorithm2(self, KH, S):
        num = KH.shape[0]
        numker = KH.shape[2]
        KH2 = np.zeros(shape=(num, num, numker))
        for p in range(numker):
            KH2_tmp = KH2[:, :, p]
            KH_tmp = KH[:, :, p]
            indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i - 1 for i in S[p]['indx']])
            KAp = KH_tmp[np.ix_(indx, indx)]
            KH2_tmp = (KAp + KAp.T) / 2
            KH2[:, :, p] = KH2_tmp

        return KH2


    def algorithm3(self, KH, S):
        num = KH.shape[0]
        numker = KH.shape[2]
        KH3 = np.zeros(shape=(num, num, numker))
        for p in range(numker):
            indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i - 1 for i in S[p]['indx']])
            n0 = len(S[p]['indx'])
            KAp = KH[np.ix_(indx, indx), p]
            KH3[np.ix_(indx, indx), p] = (KAp + KAp.T) / 2
            KH3[np.ix_(indx, [i - 1 for i in S[p]['indx'].T]), p] = np.tile(np.mean(KAp, 1), reps=(n0, 1)).T
            KH3[np.ix_([i - 1 for i in S[p]['indx'].T], indx), p] = np.tile(np.mean(KAp, 1), reps=(n0, 1))
            KH3[np.ix_([i - 1 for i in S[p]['indx'].T], [i - 1 for i in S[p]['indx']].T), p] = np.tile(
                np.mean(KH3[np.ix_([i - 1 for i in S[p]['indx']].T, indx), p], 1), reps=(n0, 1)).T

        return KH3


    def algorithm4(self, KH, S, numclass, alpha0):
        num = KH.shape[0]
        numker = KH.shape[2]
        gamma0 = np.ones(shape=(numker, 1)) / numker
        KH3 = np.zeros(shape=(num, num, numker))
        for p in range(numker):
            indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i-1 for i in S[p]['indx'].T])
            KAp = KH[np.ix_(indx, indx), p]
            KH3[np.ix_(indx, indx), p] = (KAp + KAp.T) / 2

        Kmatrix = self.my_comb_fun(KH3, gamma0**2)
        H = self.my_kernel_kmeans(Kmatrix, numclass)
        Kx = np.eye(num) - H @ H.T
        for p in range(numker):
            obs_indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i-1 for i in S[p]['indx']])
            KH3[:, :, p] = self.absent_kernel_imputation(Kx, KH3[np.ix_(obs_indx, obs_indx), p], S[p]['indx'], alpha0)

        return KH3


    def algorithm6(self, KH, S):
        num = KH.shape[0]
        numker = KH.shape[2]
        KH6 = np.zeros(shape=(num, num, numker))
        for p in range(numker):
            KH[np.ix_([i-1 for i in S[p]['indx']].T, [i-1 for i in S[p]['indx']].T), p] = math.nan
            KH6[:, :, p] = self.data_completion(KH[:, :, p], 'EM')


    def data_completion(self, X, method):
        if not np.isnan(X).any():
            raise ValueError("The missing values should be marked as NaN in the input matrix.")

        if method == 'KNN':
            imputer = KNNImputer()
            X = imputer.fit_transform(X)
        elif method == 'EM':
            imputer = IterativeImputer()  # This uses EM to fill missing values
            X = imputer.fit_transform(X)
        else:
            raise ValueError("Only KNN and EM are supported.")

        return X

    def k_center(self, K):
        n = K.shape[1]

        if np.ndim(K) == 2:
            D = np.sum(K) / n
            E = np.sum(D) / n
            J = np.ones(shape=(n, 1)) @ D
            K -= J - J.T + E @ np.ones(shape=(n, n))
            K = 0.5 * (K + K.T)

        elif np.ndim(K) == 3:
            for i in range(K.shape[2]):
                D = np.sum(K[:, :, i]) / n
                E = np.sum(D) / n
                J = np.ones(shape=(n, 1)) * D
                K[:, :, i] -= J - J.T + E * np.ones(shape=(n, n))
                K[:, :, i] = 0.5 * (K[:, :, i] + K[:, :, i].T) + 1e-12 * np.eye(n)

        return K


    def k_norm(self, K):
        if K.shape[2] > 1:
            for i in range(K.shape[2]):
                K[:, :, i] = K[:, :, i] / np.sqrt(np.diag(K[:, :, i] @ np.diag(K[:, :, i]).T))

        else:
            K = K / np.sqrt(np.diag(K) @ np.diag(K).T)

        return K


    def my_comb_fun(self, Y, gamma):
        m = Y.shape[2]
        n = Y.shape[0]
        cF = np.zeros(n)

        for p in range(m):
            cF += Y[:, :, p] * gamma[p]

        return cF


    def my_kernel_kmeans(self, K, cluster_count):
        K = (K + K.T) / 2
        _, H = eigs(K, cluster_count, which='LR')
        # obj = np.trace(H.T @ K @ H) - np.trace(K)
        H_normalized = H

        return H_normalized


    def absent_kernel_imputation(self, Kx, Kycc, mset, alpha0):
        n = Kx.shape[0]
        n0 = len(mset)
        cset = np.setdiff1d(ar1=[i for i in range(n)], ar2=mset)
        Kx0 = Kx[cset:mset, cset:mset]

        Lxmm = Kx0[n - n0 + 1:, n - n0 + 1:]
        Lxmm = (Lxmm + Lxmm.T) / 2
        Lxcm = Kx0[1:n - n0, n - n0 + 1:]

        Lxcmmm = -Lxcm / (Lxmm + alpha0 * np.eye(n0))
        Kycm = Kycc @ Lxcmmm
        Kymm = Lxcmmm.T @ Kycm
        Kyr0 = np.block(Kycc, Kycm, Kycm.T, Kymm)
        Kyr0 = (Kyr0 + Kyr0.T) / 2

        val, indxxx = np.sort([cset, mset], order='ascend')
        Kyr = Kyr0[indxxx, indxxx] + 1e-12 @ np.eye(n)

        return Kyr


    def updapte_absent_kernel_weightsV2(self, T, K, qnorm):
        num = K.shape[0]
        nb_kernel = K.shape[2]
        U0 = np.eye(num) - T @ T.T
        a = np.zeros(shape=(nb_kernel, 1))

        for p in range(nb_kernel):
            a[p] = np.trace(K[:, :, p] @ U0)

        gamma = a ** (-1 / (qnorm - 1)) / np.sum(a ** (-1 / (qnorm - 1)))
        gamma[gamma < np.finfo('float').eps] = 0
        gamma = gamma / np.sum(gamma)

        return gamma


    def cal_objV2(self, T, K, gamma0):
        nb_kernel = K.shape[2]
        num = K.shape[0]
        ZH1 = np.zeros(nb_kernel)

        for p in range(nb_kernel):
            ZH1[p, p] = (np.trace(K[:, :, p]) - np.trace(T.T @ K[:, :, p] @ T))

        obj = gamma0.T @ ZH1 * gamma0


    def my_absent_multikernel_clustering(self, K, S, cluster_count, qnorm, algorithm_choose, normalize):
        functions = {
            'algorithm0': self.algorithm0,
            'algorithm2': self.algorithm2,
            'algorithm3': self.algorithm3,
            'algorithm4': self.algorithm4,
            'algorithm6': self.algorithm6,
        }

        if normalize:
            K = self.k_center(K)
            K = self.k_norm(K)

        num = K.shape[0]
        nb_kernel = K.shape[2]
        alpha0 = 1e-3
        gamma = np.ones(shape=(nb_kernel, 1)) / nb_kernel

        if algorithm_choose in functions:
            if algorithm_choose == 'algorithm0':
                func = functions[algorithm_choose]
                KA = func(K, S, 7)
            else:
                func = functions[algorithm_choose]
                KA = func(K, S)

        KC = self.my_comb_fun(KA, gamma ** qnorm)
        flag = 1
        iter = 0
        obj = []
        while flag:
            iter += 1
            H = self.my_kernel_kmeans(KC, cluster_count)
            KA = np.zeros(shape=(num, num, nb_kernel))

            for p in range(nb_kernel):
                if len(S[p].indx) == 0:
                    KA[:, :, p] = K[:, :, p]
                else:
                    Kx = np.eye(num) - H @ H.T
                    mis_indx = [i-1 for i in S[p].indx]
                    obs_indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=mis_indx)
                    KA[:, :, p] = self.absent_kernel_imputation(Kx, K(obs_indx, obs_indx, p), mis_indx, alpha0)

            gamma = self.updapte_absent_kernel_weightsV2(H, KA, qnorm)
            obj.append(self.cal_objV2(H, KA, gamma))
            KC = self.my_comb_fun(KA, gamma ** qnorm)

            if (iter > 2) and (np.abs((obj[iter - 1] - obj[iter]) / obj[iter - 1])) < 1e-4 or (iter > 30):
                flag = 0

        H_normalized = H / np.tile(np.sqrt(np.sum(H ** 2, 1)), reps=(1, cluster_count))

        return H_normalized, gamma, obj, KA
