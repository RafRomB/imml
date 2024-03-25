import os
import numpy as np
import oct2py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels

from ..utils import check_Xs, DatasetUtils


class MKKMIF(BaseEstimator, ClassifierMixin):
    r"""
    Efficient and Effective Incomplete Multi-view Clustering (EE-IMVC).

    EE-IMVC impute missing views with a consensus clustering matrix that is regularized with prior knowledge.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    normalize : bool, default=True
        If True, it will normalize and center the kernel.
    kernel : callable, default=kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())
        Specifies the kernel type to be used in the algorithm.
    lambda_reg : float, default=1.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    qnorm : float, default=2.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
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
    H_ : array-like
        Consensus clustering matrix.
    gamma_ : array-like
        Kernel weights.
    K_ : array-like
        Kernel sub-matrix.
    loss_ : float
        Value of the loss function.

    References
    ----------
    [paper] X. Liu et al., "Multiple Kernel $k$k-Means with Incomplete Kernels," in IEEE Transactions on Pattern
            Analysis and Machine Intelligence, vol. 42, no. 5, pp. 1191-1204, 1 May 2020,
            doi: 10.1109/TPAMI.2019.2892416.
    [code]  https://github.com/wangsiwei2010/multiple_kernel_clustering_with_absent_kernel

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import MKKMIF
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> estimator = MKKMIF(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, normalize: bool = True, kernel_initialization: str = "zeros",
                 kernel: callable = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel()),
                 qnorm: float = 2., random_state: int = None, engine: str = "matlab", verbose=False):

        kernel_initializations = ['zeros', 'mean', 'knn', 'em', 'laplacian']
        if kernel_initialization not in kernel_initializations:
            raise ValueError(f"Invalid kernel_initialization. Expected one of: {kernel_initializations}")

        self.n_clusters = n_clusters
        self.normalize = normalize
        self.kernel_initialization = kernel_initialization
        self.qnorm = qnorm
        self.kernel = kernel
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose
        self.kernel_initializations = {"zeros": "algorithm2", "mean": "algorithm3", "knn": "algorithm0",
                                       "em": "algorithm6", "laplacian": "algorithm4"}

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
            matlab_folder = os.path.join("imvc", "cluster", "_mkkm_ik")
            matlab_files = ['absentKernelImputation.m', 'mycombFun.m', 'mykernelkmeans.m', 'calObjV2.m',
                            'algorithm0.m', 'algorithm2.m', 'algorithm3.m', 'algorithm4.m', 'algorithm6.m',
                            'updateabsentkernelweightsV2.m', 'myabsentmultikernelclustering.m', "kcenter.m", "knorm.m"]
            oc = oct2py.Oct2Py(temp_dir=matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    try:
                        oc.eval(f.read())
                    except:
                        print(matlab_file)

            missing_view_profile = DatasetUtils.get_missing_view_profile(Xs=Xs)
            s = [view[view == 0].index.values for _, view in missing_view_profile.items()]
            transformed_Xs = [self.kernel(X) for X in Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            transformed_Xs = np.nan_to_num(transformed_Xs, nan=0)
            s = tuple([{"indx": i} for i in s])
            kernel = self.kernel_initializations[self.kernel_initialization]

            if self.random_state is not None:
                oc.rand("seed", self.random_state)
            H_normalized,gamma,obj,KA = oc.myabsentmultikernelclustering(transformed_Xs, s, self.n_clusters,
                                                                         self.qnorm, kernel,
                                                                         int(self.normalize), nout=4)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = model.fit_predict(X=H_normalized)
        self.H_, self.gamma_, self.KA_, self.loss_ = H_normalized, gamma, KA, obj

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
