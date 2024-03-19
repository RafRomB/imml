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

    >>> from imvc.pipelines import BasePipeline
    >>> from imvc.transformers import FillIncompleteSamples, ConcatenateViews
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = BasePipeline(n_clusters = 3, transformers = [FillIncompleteSamples(value='mean'), ConcatenateViews()])
    >>> pipeline.fit_predict(Xs)
    """


    def __init__(self, estimator, n_clusters : int = None, transformers = [], random_state : int = None, memory = None,
                 verbose = False, n_jobs: int = None, **args):
        if n_clusters is not None:
            estimator.set_params(**{"n_clusters": n_clusters})
        if random_state is not None:
            estimator.set_params(**{"random_state": random_state})
        if n_jobs is not None:
            estimator.set_params(**{"n_jobs": n_jobs})
        self.estimator = estimator
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.transformers = transformers
        self.verbose = verbose
        self.memory = memory
        pipeline = make_pipeline(*transformers, estimator).set_params(**args).steps
        super(BasePipeline, self).__init__(pipeline, memory = memory, verbose = verbose)


    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return Pipeline(
                self.steps[ind], memory=self.memory, verbose=self.verbose
            )
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

