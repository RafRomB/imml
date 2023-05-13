from imvc.transformers import FillMissingViews
from sklearn.preprocessing import StandardScaler
from imvc.algorithms import MOFA
from sklearn.impute import SimpleImputer
from imvc.pipelines import BasePipeline


class MOFAPipeline(BasePipeline):
    r"""
    Firstly fill in all the missing samples with the average features for each modality, then use MOFA for data
    integration. The projected data are imputted with the mean values of each feature, stardardized and clustered
    using K-means.

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    transformers : list of transformer, default=[FillMissingViews(value="mean"), ConcatMerger(), StandardScaler()]
        The transformers to apply to the input data before clustering. Each transformer is a transformer object
        that implements the `fit_transform` method.
    **args
        Additional parameters to pass to BasePipeline.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset

    >>> from imvc.pipelines import MOFAPipeline
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = MOFAPipeline(n_clusters = 3).fit(Xs)
    >>> labels = pipeline.predict(Xs)
    """

    def __init__(self, n_clusters: int = None,
                 transformers=[FillMissingViews(value="nan"), MOFA().set_output(transform='pandas'),
                               SimpleImputer(strategy='mean').set_output(transform='pandas'),
                               StandardScaler().set_output(transform='pandas')], **args):
        self.n_clusters = n_clusters
        self.transformers = transformers
        super().__init__(n_clusters=n_clusters, transformers=transformers, **args)
