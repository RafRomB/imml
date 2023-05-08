from mvlearn.compose import ConcatMerger
from imvc.pipelines import BasePipeline
from imvc.transformers import FillMissingViews
from sklearn.preprocessing import StandardScaler


class ConcatPipeline(BasePipeline):
    r"""
    Firstly fill in all the missing samples with the average features for each modality, and then concatenate all
    modal features into one. A clustering with K-means is performed then.

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
    >>> pipeline = ConcatPipeline(n_clusters = 3)
    >>> pipeline.fit_predict(Xs)
    """

    
    def __init__(self, n_clusters : int = None,
                 transformers = [FillMissingViews(value="mean"), ConcatMerger(), StandardScaler().set_output(transform = 'pandas')], **args):
        self.n_clusters = n_clusters
        self.transformers = transformers
        super().__init__(n_clusters = n_clusters, transformers = transformers, **args)
