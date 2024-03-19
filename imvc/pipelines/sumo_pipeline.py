from sklearn.preprocessing import StandardScaler

from ..algorithms import SUMO
from . import BasePipeline
from ..transformers import SortData, MultiViewTransformer, FillIncompleteSamples


class SUMOPipeline(BasePipeline):
    r"""
    The pipeline comprises a data sortening, missing sample filling with nan, standardization of data and SUMO algorithm.

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to generate.
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
    >>> from imvc.pipelines import SUMOPipeline
    >>> Xs = LoadDataset.load_incomplete_digits(p = 0.2)
    >>> pipeline = SUMOPipeline(n_clusters=2)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters = None, memory=None, verbose=False, random_state : int = None, **args):
        estimator = SUMO(n_clusters= n_clusters, random_state=random_state)
        transformers = [SortData(), FillIncompleteSamples(value="mean"),
                        MultiViewTransformer(transformer=StandardScaler().set_output(transform='pandas'))]
        super().__init__(estimator=estimator, transformers=transformers, memory=memory, verbose=verbose,
                         n_clusters=estimator.n_clusters, random_state=random_state, **args)
