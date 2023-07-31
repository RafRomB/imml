from sklearn.cluster import KMeans
from imvc.pipelines import BasePipeline
from imvc.transformers import FillMissingViews, SortData, ConcatenateViews
from sklearn.preprocessing import StandardScaler


class ConcatPipeline(BasePipeline):
    r"""
    This pipeline assess the order of the dataset firstly. Then fill in all the missing samples with the average
    features for each modality, and then concatenate all modal features into one. A clustering with K-means
    is performed finally.

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

    References
    ----------
    [paper] Handong Zhao, Hongfu Liu, and Yun Fu. 2016. Incomplete multi-modal visual data grouping. In
        International Joint Conferences on Artificial Intelligence. 2392--2398.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.pipelines import ConcatPipeline
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = ConcatPipeline(n_clusters = 3)
    >>> labels = pipeline.fit_predict(Xs)
    """

    
    def __init__(self, n_clusters : int = None, memory=None, verbose=False, random_state : int = None, **args):

        estimator = KMeans(n_clusters = n_clusters, random_state=random_state)
        transformers = [SortData(), FillMissingViews(value="mean"), ConcatenateViews(), StandardScaler().set_output(transform='pandas')]
        super().__init__(estimator = estimator, transformers = transformers, memory = memory, verbose = verbose,
                         n_clusters=n_clusters, random_state=random_state, **args)
