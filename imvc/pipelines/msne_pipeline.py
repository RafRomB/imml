from sklearn.preprocessing import StandardScaler

from algorithms import MSNE
from pipelines import BasePipeline
from transformers import SortData, MultiViewTransformer


class MSNEPipeline(BasePipeline):
    r"""
    The pipeline comprises a standardization of data followed by the MSNE algorithm. In addition, the first step is
    data sortening.

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
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
    >>> from imvc.pipelines import MONETPipeline
    >>> Xs = LoadDataset.load_incomplete_digits(p = 0.2)
    >>> pipeline = MONETPipeline()
    >>> pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = None, memory=None, verbose=False, **args):
        estimator = MSNE(n_clusters=n_clusters)
        transformers = [SortData(), MultiViewTransformer(transformer=StandardScaler().set_output(transform='pandas'))]
        super().__init__(estimator=estimator, transformers=transformers, memory=memory, verbose=verbose, **args)
