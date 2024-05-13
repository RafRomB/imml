from . import BasePipeline
from ..preprocessing import ConvertToNM, FillIncompleteSamples, MultiViewTransformer, ConcatenateViews, SortData
from ..algorithms import NMFC


class NMFCPipeline(BasePipeline):
    r"""
    Firstly fill in all the missing samples with the average features for each modality, then convert each view into
    a non-negative matrix and finally apply a NMF for the clustering.

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    random_state : int (default=None)
        Determines the randomness. Use an int to make the randomness deterministic.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted preprocessing of the pipeline. By default, no caching is performed. If a string is
        given, it is the path to the caching directory. Enabling caching triggers a clone of the preprocessing before
        fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly. Use the
        attribute named_steps or steps to inspect estimators within the pipeline. Caching the preprocessing is
        advantageous when fitting is time consuming.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it is completed.
    **args
        Additional parameters to pass to the pipeline.

    References
    ----------
    Several papers used this strategy with successful results, such as:
    [paper] Cao, L. et al. Proteogenomic characterization of pancreatic ductal adenocarcinoma. Cell 184,
        5031–5052 e5026 (2021).
    [url] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.pipelines import NMFCPipeline
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = NMFCPipeline(n_clusters = 3)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = None, memory=None, verbose=False, random_state : int = None, **args):
        estimator = NMFC(n_components=n_clusters, random_state=random_state).set_output(transform='pandas')
        transformers = [FillIncompleteSamples(value="mean"), MultiViewTransformer(ConvertToNM()), ConcatenateViews()]
        super().__init__(estimator=estimator, transformers=transformers, memory=memory, verbose=verbose,
                         random_state=random_state, **args)
