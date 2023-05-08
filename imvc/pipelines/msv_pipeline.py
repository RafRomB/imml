from sklearn.impute import SimpleImputer
from imvc.pipelines import MultiViewPipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imvc.transformers import AddMissingViews


class MSVPipeline(MultiViewPipeline):
    r"""
    Firstly fill in all the missing samples with the average features for each modality, and then cluster with K-means.

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    transformers : list of transformer, default=[FillMissingViews(value="mean"), ConcatMerger(), StandardScaler()]
        The transformers to apply to the input data before clustering. Each transformer is a transformer object
        that implements the `fit_transform` method.
    **args
        Additional parameters to pass to BasePipeline.

    References
    ----------
    [paper] Handong Zhao, Hongfu Liu, and Yun Fu. 2016. Incomplete multi-modal visual data grouping. In
        International Joint Conferences on Artificial Intelligence. 2392--2398.

    Examples
    --------
    >>> from imvc.datasets import load_incomplete_nutrimouse
    >>> from imvc.pipelines import ConcatPipeline
    >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
    >>> pipeline = MSVPipeline()
    >>> pipeline.fit_predict(Xs)
    """


    def __init__(self, view_estimators = KMeans(), n_clusters : int = None,
                 view_transformers = [AddMissingViews(samples=None),
                                      SimpleImputer(strategy = 'mean').set_output(transform = 'pandas'),
                                      StandardScaler().set_output(transform = 'pandas')],
                 samples = None, memory = None, verbose = False, **kwargs):
        if samples is not None:
            view_transformers[0].set_params(**{"samples": samples})
        self.n_clusters = n_clusters
        self.samples = samples
        self.view_estimators = view_estimators
        self.view_transformers = view_transformers
        self.verbose = verbose
        self.memory = memory
        self.kwargs = kwargs

        if not isinstance(view_transformers, list):
            view_transformers = [view_transformers]
        if not isinstance(view_estimators, list):
            view_estimators = [view_estimators]
        if n_clusters is not None:
            view_estimators = [estimator.set_params(**{"n_clusters": n_clusters}) for estimator in view_estimators]
        self.same_transformers_ = False if isinstance(view_transformers[0], list) else True
        self.same_estimator_ = False if isinstance(view_estimators[0], list) else True
        if self.same_transformers_:
            if self.same_estimator_:
                self.steps_ = view_transformers + view_estimators
            else:
                self.steps_ = [view_estimators + i for i in view_estimators]
        else:
            if self.same_estimator_:
                self.steps_ = [i + view_estimators for i in view_transformers]
            else:
                self.steps_ = [i + j for i,j in zip(view_transformers, view_estimators)]

        super().__init__(steps = self.steps_, memory = memory, verbose = verbose, **kwargs)

