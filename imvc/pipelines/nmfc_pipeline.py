from mvlearn.compose import ConcatMerger
from imvc.pipelines import BasePipeline
from imvc.transformers import ConvertToNM, FillMissingViews, MultiViewTransformer
from imvc.modules import NMFC


class NMFCPipeline(BasePipeline):
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
    >>> from imvc.pipelines import NMFCPipeline
    >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
    >>> pipeline = NMFCPipeline(n_clusters = 3)
    >>> pipeline.fit_predict(Xs)
    """

    
    def __init__(self, estimator = NMFC().set_output(transform = 'pandas'), n_clusters : int = None,
                 transformers = [FillMissingViews(value="mean"),
                                 MultiViewTransformer(ConvertToNM()),
                                 ConcatMerger()], **args):
        if n_clusters is not None:
            estimator.set_params(**{"n_components": n_clusters})
        self.n_clusters = n_clusters
        self.estimator = estimator
        self.transformers = transformers
        super().__init__(estimator = estimator, transformers = transformers, **args)
