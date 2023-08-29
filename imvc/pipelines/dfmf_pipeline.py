from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from imvc.pipelines import BasePipeline
from imvc.transformers import SortData, FillMissingViews, MultiViewTransformer, DFMF


class DFMFPipeline(BasePipeline):
    r"""
    Sort the dataset, fill in all the missing samples with the average features for each modality and use MOFA for data
    integration. The projected data are imputted with the mean values of each feature, stardardized and clustered
    using K-means.

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    random_state : int (default=None)
        Determines the randomness. Use an int to make the randomness deterministic.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default, no caching is performed. If a string is
        given, it is the path to the caching directory. Enabling caching triggers a clone of the transformers before
        fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly. Use the
        attribute named_steps or steps to inspect estimators within the pipeline. Caching the transformers is
        advantageous when fitting is time consuming.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it is completed.
    **args
        Additional parameters to pass to the pipeline.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.pipelines import MOFAPipeline
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = MOFAPipeline(n_clusters = 3).fit(Xs)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = None, memory=None, verbose=False, random_state : int = None, **args):
        estimator = KMeans(n_clusters = n_clusters, random_state=random_state)
        transformers = [SortData(), FillMissingViews(value="nan"),
                        MultiViewTransformer(transformer=StandardScaler().set_output(transform='pandas')),
                        # DFMF(),
                        DFMF().set_output(transform='pandas'),
                        StandardScaler().set_output(transform='pandas')]
        super().__init__(estimator = estimator, transformers = transformers, memory = memory, verbose = verbose,
                         n_clusters=n_clusters, random_state=random_state, **args)
