import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import Utils, check_Xs


class jNMF(TransformerMixin, BaseEstimator):
    r"""
    jNMF (Joint Non-negative Matrix Factorization Algorithms).

    jNMF decompose the matrices to two low-dimensional factor matrices. It can deal with both view- and
    single-wise missing. It cannot transform new data.

    R library "nnTensor" should be installed. This can be done using the R command: install.packages("nnTensor")

    Parameters
    ----------
    n_components : int, default=10
        Number of components to keep.
    max_iter : int, default=100
        Maximum number of iterations to perform.
    init_type : str or list of str, default='random_c'
        The algorithm to initialize latent matrix factors. Options are 'random', 'random_c' and 'random_vcol'. It can be
        a list, each item being for fit and transform, respectively.
    n_run: int, default=1
        Number of components to keep.
    stopping : tuple (target_matrix, eps), default=None
        Terminate iteration if the reconstruction error of target matrix improves by less than eps.
    stopping_system : float, default=None
        Terminate iteration if the reconstruction error of the fused system improves by less than eps. compute_err is
        to True to compute the error of the fused system.
    compute_err : bool, default=False
        Compute the reconstruction error of every relation matrix if True.
    callback : callable, default=None
        An optional user-supplied function to call after each iteration. Called as callback(G, S, cur_iter), where
        S and G are current latent estimates.
    fill_value : float, default=0
        Value to use to initially fill missing values.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors.

    Attributes
    ----------
    fuser_ : Dfmf object
        Model.
    transformer_ : DfmfTransform object
        Object for transforming unseen data.
    t_: fusion.ObjectType
    ts_: list of fusion.ObjectType

    References
    ----------
    [paper1] Liviu Badea, (2008) Extracting Gene Expression Profiles Common to Colon and Pancreatic Adenocarcinoma
            using Simultaneous nonnegative matrix factorization. Pacific Symposium on Biocomputing 13:279-290.
    [paper2] Shihua Zhang, et al. (2012) Discovery of multi-dimensional modules by integrative analysis of cancer
            genomic data. Nucleic Acids Research 40(19), 9379-9391.
    [paper3] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in heterogeneous
            omics multi-modal data, Bioinformatics 32(1), 1-8.
    [paper4] Y. Kenan Yilmaz et al., (2010) Probabilistic Latent Tensor Factorization, International Conference on
            Latent Variable Analysis and Signal Separation 346-353.
    [paper5] N. Fujita et al., (2018) Biomarker discovery by integrated joint non-negative matrix factorization and
            pathway signature analyses, Scientific Report.
    [code] https://rdrr.io/cran/nnTensor/man/jNMF.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.decomposition import jNMF
    >>> from imvc.transformers import MultiViewTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.cluster import KMeans
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> transformer = jNMF(n_components = 5).set_output(transform="pandas")
    >>> estimator = KMeans(n_clusters = 3)
    >>> pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")), transformer, estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """


    def __init__(self, n_components : int = 10, init_w = None, init_v = None, init_h = None, fix_w: bool = False,
                 fix_v: bool = False, fix_h: bool =False, l1_w: float = 1e-10, l1_v: float = 1e-10, l1_h: float = 1e-10,
                 l2_w: float = 1e-10, l2_v: float = 1e-10, l2_h: float = 1e-10, w = None,
                 algorithm = ["Frobenius", "KL", "IS", "PLTF"], p: float = 1., thr: float = 1e-10, num_iter: int = 100,
                 verbose=0, random_state: int = None):
        self.n_components = n_components
        self.init_w = init_w
        self.init_v = init_v
        self.init_h = init_h
        self.fix_w = fix_w
        self.fix_v = fix_v
        self.fix_h = fix_h
        self.l1_w = l1_w
        self.l1_v = l1_v
        self.l1_h = l1_h
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_h = l2_h
        self.w = w
        self.algorithm = algorithm
        self.p = p
        self.thr = thr
        self.num_iter = num_iter
        self.verbose = bool(verbose)
        self.random_state = random_state
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
        samples = Xs[0].index
        mask = [X.notnull().astype(int) for X in Xs]
        transformed_Xs, transformed_mask = Utils._convert_df_to_r_object(Xs), Utils._convert_df_to_r_object(mask)
        if self.algorithm is not None:
            algorithm = ro.vectors.StrVector(self.algorithm)
        base, nnTensor = importr("base"), importr("nnTensor")
        if self.random_state is not None:
            base.set_seed(self.random_state)
        init_w = ro.NULL if self.init_w is None else self.init_w
        init_v = ro.NULL if self.init_v is None else self.init_v
        init_h = ro.NULL if self.init_h is None else self.init_h
        w = ro.NULL if self.w is None else self.w

        w, v, h, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
            X= transformed_Xs, M=transformed_mask, J=self.n_components,
            initW=init_w, initV=init_v, initH=init_h,
            fixW=self.fix_w, fixV=self.fix_v, fixH=self.fix_h,
            L1_W=self.l1_w, L1_V=self.l1_v, L1_H=self.l1_h,
            L2_W=self.l2_w, L2_V= self.l2_v, L2_H=self.l2_h,
            w=w, algorithm=algorithm, p=self.p, thr = self.thr, num_iter=self.num_iter, verbose=self.verbose)

        v = [np.array(mat) for mat in v]
        h = [np.array(mat) for mat in h]
        if self.transform_ == "pandas":
            w = pd.DataFrame(np.array(w), index= samples)
            v, h = [pd.DataFrame(mat, index= samples) for mat in v], [pd.DataFrame(mat) for mat in h]

        self.w_, self.v_, self.h_ = w, v, h
        self.reconstruction_err_ = list(recerror)
        self.train_reconstruction_err_ = list(train_recerror)
        self.test_reconstruction_err_ = list(test_recerror)
        self.relchange_ = list(relchange)
        return self


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
        transformed_Xs : list of array-likes, shape (n_samples, n_features)
            The projected data.
        """
        #todo
        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        transformed_X = self.w_
        return transformed_X


    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self
