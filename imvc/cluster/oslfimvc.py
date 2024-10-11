import os
from os.path import dirname
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels
from scipy.sparse.linalg import eigs
from scipy.linalg import orth

from ..impute import get_observed_view_indicator
from ..utils import check_Xs

try:
    import oct2py
    oct2py_installed = True
except ImportError:
    oct2py_installed = False
    oct2py_module_error = "Oct2Py needs to be installed to use matlab engine."


class OSLFIMVC(BaseEstimator, ClassifierMixin):
    r"""
    One-Stage Incomplete Multi-view Clustering via Late Fusion (OS-LF-IMVC).

    OS-LF-IMVC integrates the processes of imputing incomplete views and clustering into a cohesive optimization
    procedure. This approach enables the direct utilization of the learned consensus partition matrix to enhance
    the final clustering task.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    kernel : callable, default=kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())
        Specifies the kernel type to be used in the algorithm.
    lambda_reg : float, default=1.
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
    embedding_ : np.array
        Consensus clustering matrix to be used as input for the KMeans clustering step.
    WP_ : array-like of shape (n_clusters, n_clusters, n_views)
        p-th permutation matrix.
    C_ : array-like of shape (n_clusters, n_clusters)
        Centroids.
    beta_ : array-like of shape (n_views,)
        Adaptive weights of clustering matrices.
    loss_ : array-like of shape (n_iter_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#oslfimvcpaper] Yi Zhang, Xinwang Liu, Siwei Wang, Jiyuan Liu, Sisi Dai, and En Zhu. 2021. One-Stage Incomplete
                        Multi-view Clustering via Late Fusion. In Proceedings of the 29th ACM International Conference
                        on Multimedia (MM '21). Association for Computing Machinery, New York, NY, USA, 2717–2725.
                        https://doi.org/10.1145/3474085.3475204.
    .. [#oslfimvccode] https://github.com/ethan-yizhang/OSLF-IMVC

    Example
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import OSLFIMVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from imvc.preprocessing import MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = StandardScaler().set_output(transform="pandas")
    >>> estimator = OSLFIMVC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)

    """

    def __init__(self, n_clusters: int = 8, kernel: callable = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel()),
                 lambda_reg: float = 1., random_state:int = None, engine: str ="matlab", verbose = False):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not oct2py_installed):
            raise ModuleNotFoundError(oct2py_module_error)

        self.n_clusters = n_clusters
        self.kernel = kernel
        self.lambda_reg = lambda_reg
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
            self._oc.eval("pkg load statistics")


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
            observed_view_indicator = get_observed_view_indicator(Xs)
            if isinstance(observed_view_indicator, pd.DataFrame):
                observed_view_indicator = observed_view_indicator.reset_index(drop=True)
            elif isinstance(observed_view_indicator[0], np.ndarray):
                observed_view_indicator = pd.DataFrame(observed_view_indicator)
            s = [view[view == 0].index.values for _,view in observed_view_indicator.items()]
            transformed_Xs = [self.kernel(X) for X in Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            s = tuple([{"indx": i +1} for i in s])

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            U, C, WP, beta, obj = self._oc.OS_LF_IMVC_alg(transformed_Xs, s, self.n_clusters, self.lambda_reg, nout=5)
            beta = beta[:,0]
            obj = obj[0]

        elif self.engine=="python":
            observed_view_indicator = get_observed_view_indicator(Xs)
            if isinstance(observed_view_indicator, pd.DataFrame):
                observed_view_indicator = observed_view_indicator.reset_index(drop=True)
            elif isinstance(observed_view_indicator[0], np.ndarray):
                observed_view_indicator = pd.DataFrame(observed_view_indicator)
            s = [view[view == 0].index.values for _, view in observed_view_indicator.items()]
            transformed_Xs = [self.kernel(X) for X in Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            s = tuple([{"indx": i + 1} for i in s])

            if self.random_state is not None:
                np.random.seed(self.random_state)
            U, C, WP, beta, obj = self.oslfimvc(transformed_Xs, s, self.n_clusters, self.lambda_reg)

        model = KMeans(n_clusters= self.n_clusters, n_init= "auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X= U)
        self.embedding_, self.WP_, self.C_, self.beta_, self.loss_ = U, WP, C, beta, obj
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


    def initialiaze_kh(self, KH, S):
        r"""
        Initialize KH variable

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, n_views)
        S: tuple of shape (n_views)
            - S[i]['indx']: list of numbers representing missing values column.

        Returns
        -------
        KH: 3-D array of shape(n_samples, n_samples, n_views)
        """
        numker = KH.shape[2]
        num = KH.shape[0]
        for p in range(numker):
            KH_tmp = KH[:, :, p]
            mis_set = [i-1 for i in S[p]['indx']]
            obs_set = np.setdiff1d(ar1=[i for i in range(num)], ar2=mis_set)
            WP = np.zeros(shape=(len(obs_set), len(mis_set)))
            KH_tmp[np.ix_(obs_set, obs_set)] = KH_tmp[np.ix_(obs_set, obs_set)]
            KH_tmp[np.ix_(obs_set, mis_set)] = np.matmul(KH_tmp[np.ix_(obs_set, obs_set)], WP)
            KH_tmp[np.ix_(mis_set, obs_set)] = KH_tmp[np.ix_(obs_set, mis_set)].T
            KH_tmp[np.ix_(mis_set, mis_set)] = np.matmul(KH_tmp[np.ix_(mis_set, obs_set)], WP)
            KH[:, :, p] = KH_tmp

        return KH


    def my_comb_fun(self, Y, beta):
        r"""
        Combine base kernel.

        Parameters
        ----------
        Y: array of shape (n_samples, n_clusters)
        beta: list of floats of length (n_views)

        Returns
        -------
        cF: array of shape (n_samples, n_clusters)
        """
        m = Y.shape[2]
        n = Y.shape[0]
        cF = np.zeros(shape=(n, n))

        for p in range(m):
            cF += Y[:, :, p] * beta[p]
        return cF


    def my_initialization(self, KH, S, n_clusters):
        r"""
        Initialize HP and WP variable

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, n_views)
        S: tuple of shape (n_views)
            - S[i]['indx']: list of numbers representing missing values column.
        n_clusters : int, default=8
            The number of clusters to generate.

        Returns
        -------
        HP: 3-D array of shape (n_samples, n_clusters, n_views)
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        """
        numker = KH.shape[2]
        num = KH.shape[0]
        HP = np.zeros(shape=(num, n_clusters, numker))
        WP = np.zeros(shape=(n_clusters, n_clusters, numker))

        for p in range(numker):
            obs_indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i-1 for i in S[p]['indx']])
            KH_temp = KH[:, :, p]
            KAp = KH_temp[np.ix_(obs_indx, obs_indx)]
            KAp = ((KAp + KAp.T)/2) + (1e-8 * np.eye(len(obs_indx)))
            _, Hp = eigs(A=KAp, k=n_clusters, which='LR')

            HP_tmp = HP[:, :, p]
            HP_tmp[np.ix_(obs_indx), :] = Hp
            HP[:, :, p] = HP_tmp

            WP[:, :, p] = np.eye(n_clusters)

        return HP, WP


    def my_initialization_C(self, KH, n_clusters):
        r"""
        Initialize C variable.

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, n_views)
        n_clusters : int, default=8
            The number of clusters to generate.

        Returns
        -------
        C: array of shape (n_clusters, n_clusters)
        """
        numker = KH.shape[2]
        sigma0 = np.ones(shape=(numker, 1)) / numker
        avgker = self.my_comb_fun(KH, sigma0)  # Débuggé
        H_normalized1 = self.my_kernel_kmeans(avgker, n_clusters)
        H_normalized1 = H_normalized1 / np.tile(np.sqrt(np.sum(H_normalized1**2, 1)), reps=(n_clusters, 1)).T
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, max_iter=200, n_init=30).fit(np.real(H_normalized1))
        C = orth(kmeans.cluster_centers_)
        return C


    def my_kernel_kmeans(self, K, n_clusters):
        r"""
        Determines eigenvectors.

        Parameters
        ----------
        K: 2-D array of shape (n_samples, n_samples)
        n_clusters : int, default=8
            The number of clusters to generate.

        Returns
        -------
        H_normalized: 2-D array of shape (n_samples, n_clusters)
        """
        K = (K + K.T) / 2
        _, H = eigs(A=K, k=n_clusters, which='LR')
        H_normalized = H

        return H_normalized


    def my_solving(self, B):
        r"""
        Optimize Y.

        Parameters
        ----------
        B: array of shape (n_samples, n_clusters)

        Returns
        -------
        Y: array of shape (n_samples, n_clusters)
        """
        num, k = B.shape
        I = np.argmax(B, axis=1)
        Y = np.zeros(shape=(num, k))
        for i in range(num):
            Y[i, I[i]] = 1
        return Y


    def update_beta_oslfimvc(self, HP, WP, Y, C):
        r"""
        Update beta variable.

        Parameters
        ----------
        HP: 3-D array of shape (n_samples, n_clusters, n_views)
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        Y: array of shape (n_samples, n_clusters)
        C: array of shape (n_clusters, n_clusters)

        Returns
        -------
        beta: list of floats of length (n_views)
        """
        numker = WP.shape[2]
        HHPWP = np.zeros(shape=(numker, 1))
        Hstar = np.matmul(Y, C)
        for p in range(numker):
            HHPWP[p] = np.trace(np.matmul(Hstar.T, (np.matmul(HP[:, :, p], WP[:, :, p]))))
        beta = HHPWP / np.linalg.norm(HHPWP)
        beta[beta < np.finfo(float).eps] = 0
        beta = beta / np.linalg.norm(beta)
        return beta


    def update_wp_oslfimvc(self, HP, Y, C):
        r"""
        Update WP variable.

        Parameters
        ----------
        HP: 3-D array of shape (n_samples, n_clusters, n_views)
        Y: array of shape (n_samples, n_clusters)
        C: array of shape (n_clusters, n_clusters)

        Returns
        -------
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        """
        k = HP.shape[1]
        numker = HP.shape[2]
        WP = np.zeros(shape=(k, k, numker))
        Hstar = np.matmul(Y, C)
        for p in range(numker):
            TP = np.matmul(HP[:, :, p].T, Hstar)
            Up, Sp, Vp = svd(TP, full_matrices=False)
            WP[:, :, p] = np.matmul(Up, Vp.T.conj().T)
        return WP

    def oslfimvc(self, KH, S, n_clusters, lambda_reg):
        r"""
        Runs OSLFIMVC clustering algorithm.

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, n_views)
        S: tuple of shape (n_views)
            - S[i]['indx']: list of numbers representing missing values column.
        n_clusters : int, default=8
            The number of clusters to generate.
        lambda_reg : float, default=1.
            Regularization parameter. The algorithm demonstrated stable performance across a wide range of
            this hyperparameter.

        Returns
        -------
        Y: array of shape (n_samples, n_clusters)
        C: array of shape (n_clusters, n_clusters)
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        beta: list of floats of length (n_views)
        obj: list of floats
            The length of this list is the number of iterations.
        """
        num = KH.shape[1]
        numker = KH.shape[2]
        max_iter = 100
        KH = self.initialiaze_kh(KH, S)  # Débuggé
        HP, WP = self.my_initialization(KH, S, n_clusters)  # Débuggé
        HP00 = HP
        beta = np.ones(shape=(numker, 1)) * np.sqrt(1 / numker)
        C = self.my_initialization_C(KH, n_clusters)  # Débuggé
        KC = self.my_comb_fun(KH, beta)  # Débuggé
        KC = (KC + KC.T) / 2
        H0 = self.my_kernel_kmeans(KC, n_clusters)

        flag = 1
        iter = 0

        obj = []
        obj.append(0)

        RpHpwp = np.zeros(shape=(num, n_clusters))
        for p in range(numker):
            RpHpwp += beta[p] * np.matmul(HP[:, :, p], WP[:, :, p])

        RpHpwp_lambda = RpHpwp + (lambda_reg * H0)
        while flag:
            iter += 1
            # First step : optimize Y with given (WP, C, beta)
            YB = np.matmul(RpHpwp_lambda, C.T)
            Y = self.my_solving(YB)
            # Second step : otpimize C with given (Y, WP, beta)
            CB = np.matmul(Y.T, RpHpwp_lambda)
            Uh, Sh, Vh = np.linalg.svd(CB, full_matrices=False)
            V = Vh.T.conj()
            C = np.matmul(Uh, V.T)

            # Third step : optimize WP with given (C, Y, beta)
            WP = self.update_wp_oslfimvc(HP, Y, C)

            # Fourth step : optimize HP with given (WP, Y, C and beta)
            for p in range(numker):
                mis_set = [i-1 for i in S[p]['indx']]
                obs_set = np.setdiff1d(ar1=[i for i in range(num)], ar2=mis_set)
                HB = np.matmul(np.matmul(Y[mis_set, :], C), WP[:, :, p].T)
                Uh, Sh, Vh = np.linalg.svd(HB, full_matrices=False)
                V = Vh.T.conj()
                if len(mis_set) > 0:
                    HP[mis_set, :, p] = np.matmul(Uh, V.T)
                HP[obs_set, :, p] = HP00[obs_set, :, p]

            # Fifth step : optimize beta with given (HP, WP, Y, C)
            beta = self.update_beta_oslfimvc(HP, WP, Y, C)

            RpHpwp = np.zeros(shape=(num, n_clusters))
            for p in range(numker):
                RpHpwp += beta[p] * np.matmul(HP[:, :, p], WP[:, :, p])
            RpHpwp_lambda = RpHpwp + (lambda_reg * H0)

            obj.append(np.trace(np.matmul(np.matmul(Y, C).T, RpHpwp_lambda)))
            if (iter > 2) and ((np.abs((obj[iter] - obj[iter - 1]) / obj[iter]) < 1e-4) or (iter > max_iter)):
                flag = 0

            Y = Y / np.tile(A=np.sqrt(np.sum(Y ** 2, 1)), reps=(n_clusters, 1)).T
            return Y, C, WP, beta, obj
