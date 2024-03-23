import numpy as np
from rpy2.robjects.packages import importr

from ..utils import Utils, check_Xs


class JNMF(object):
    r"""
    jNMF (Joint Non-negative Matrix Factorization Algorithms).

    jNMF decompose the matrices to two low-dimensional factor matrices. It can deal with both view- and
    single-wise missing.

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
    [paper1] Hong-Qiang Wang, Chun-Hou Zheng, Xing-Ming Zhao, jNMFMA: a joint non-negative matrix factorization
            meta-analysis of transcriptomics data, Bioinformatics, Volume 31, Issue 4, February 2015, Pages 572–580,
            https://doi.org/10.1093/bioinformatics/btu679.
    [paper2] Liviu Badea, (2008) Extracting Gene Expression Profiles Common to Colon and Pancreatic Adenocarcinoma
            using Simultaneous nonnegative matrix factorization. Pacific Symposium on Biocomputing 13:279-290.
.
    [paper3] Shihua Zhang, et al. (2012) Discovery of multi-dimensional modules by integrative analysis of cancer
            genomic data. Nucleic Acids Research 40(19), 9379-9391.
.
    [paper4] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in heterogeneous
            omics multi-modal data, Bioinformatics 32(1), 1-8.
.
    [paper5] Y. Kenan Yilmaz et al., (2010) Probabilistic Latent Tensor Factorization, International Conference on
            Latent Variable Analysis and Signal Separation 346-353.
.
    [paper6] N. Fujita et al., (2018) Biomarker discovery by integrated joint non-negative matrix factorization and
            pathway signature analyses, Scientific Report.
    [code] https://rdrr.io/cran/nnTensor/man/jNMF.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.decomposition import DFMF
    >>> from imvc.transformers import MultiViewTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.cluster import KMeans
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> dfmf = DFMF(n_components = 5).set_output(transform="pandas")
    >>> estimator = KMeans(n_clusters = 3)
    >>> pipeline = make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")), dfmf, estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """


    def __init__(self, n_components : int = 10, init_w = np.nan, init_v = np.nan, init_h = np.nan, fix_w: bool = False,
                 fix_v: bool = False, fix_h: bool =False, l1_w: float = 1e-10, l1_v: float = 1e-10, l1_h: float = 1e-10,
                 l2_w: float = 1e-10, l2_v: float = 1e-10, l2_h: float = 1e-10, w = np.nan, algorithm: str = np.nan,
                 p: float = 1., thr: float = 1e-10, num_iter: int = 100, verbose=0, random_state: int = None):
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
        self.verbose = verbose
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
        mask = [X.notnull().astype(int) for X in Xs]
        Xs, mask = Utils.convert_df_to_r_object(Xs), Utils.convert_df_to_r_object(mask)
        base, nnTensor = importr("base"), importr("nnTensor")
        base.set_seed(self.random_state)
        w, v, h, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
            X= Xs, M=mask, initW=self.init_w, initV=self.init_v, initH=self.init_h, fixW=self.fix_w, fixV=self.fix_v,
            fixH=self.fix_h, L1_W=self.l1_w, L1_V=self.l1_v, L1_H=self.l1_h, L2_W=self.l2_w, L2_V= self.l2_v,
            L2_H=self.l2_h, J=self.n_components, w=self.w, algorithm=self.algorithm, p=self.p, thr = self.thr,
            num_iter=self.num_iter, verbose=self.verbose)
        self.v_, self.relchange_ = v, relchange
        self.reconstruction_err_, self.train_reconstruction_err_, self.test_reconstruction_err_ = recerror, train_recerror, test_recerror
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
        relations = [fusion.Relation(X.values, self.t_, t) for X,t in zip(Xs, self.ts_)]
        fusion_graph = fusion.FusionGraph(relations)
        transformed_X = self.transformer_.transform(self.t_, fusion_graph, self.fuser_).factor(self.t_)
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index= Xs[0].index)
        return transformed_X


    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self
