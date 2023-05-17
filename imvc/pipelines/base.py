from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


class BasePipeline(Pipeline):
    r"""
    A base pipeline that applies a sequence of transformers followed by an estimator for clustering.

    Parameters
    ----------
    estimator : estimator object
        The estimator to use for clustering.
    n_clusters : int, default=None
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    transformers : list of transformer, default=[]
        The transformers to apply to the input data before clustering. Each transformer is a transformer object
        that implements the `fit_transform` method.
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

    >>> from mvlearn.compose import ConcatMerger
    >>> from imvc.pipelines import BasePipeline
    >>> from imvc.transformers import FillMissingViews
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = BasePipeline(n_clusters = 3, transformers = [FillMissingViews(value='mean'), ConcatMerger()])
    >>> pipeline.fit_predict(Xs)
    """


    def __init__(self, estimator, n_clusters : int = None, transformers = [], memory = None,
                 verbose = False, **args):
        if n_clusters is not None:
            estimator.set_params(**{"n_clusters": n_clusters})
        self.estimator = estimator
        self.n_clusters = n_clusters
        self.transformers = transformers
        self.verbose = verbose
        self.memory = memory
        super().__init__(make_pipeline(*transformers, estimator).set_params(**args).steps,
                         memory = memory, verbose = verbose)
