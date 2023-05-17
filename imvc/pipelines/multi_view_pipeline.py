from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline


class MultiViewPipeline(BaseEstimator, ClassifierMixin):
    r"""
    An estimator that applies the same pipeline to multiple views of data.

    Parameters
    ----------
    steps : list of Estimator objects or list of list of Estimator objects
        List of the scikit-learn estimators that are chained together to estimate each view of data. If a list of list
        is provided, each pipeline will be applied on each view, otherwise the same pipeline will be applied on each view.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default, no caching is performed. If a string is
        given, it is the path to the caching directory. Enabling caching triggers a clone of the transformers before
        fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly. Use the
        attribute named_steps or steps to inspect estimators within the pipeline. Caching the transformers is
        advantageous when fitting is time consuming.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it is completed.

    Attributes
    ----------
    pipeline_list_ : list of pipeline (n_views,)
        A list of pipelines, one for each view of data.
    same_pipeline_ : boolean
        A booleaing indicating if the same pipeline will be applied on each view of data.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.pipelines import MultiViewPipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.cluster import KMeans
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> mv_pipeline = MultiViewPipeline(steps = [StandardScaler(), KMeans(n_clusters=3)])
    >>> labels = mv_pipeline.fit_predict(Xs)
    """

    def __init__(self, steps: list, memory = None, verbose = False, **kwargs):
        self.steps = steps
        self.verbose = verbose
        self.memory = memory
        self.kwargs = kwargs
        self.same_pipeline_ = False if isinstance(steps[0], list) else True
        if self.same_pipeline_:
            self.pipeline_list_ = []
        else:
            self.pipeline_list_ = [make_pipeline(*steps_idx, memory = memory, verbose = verbose).set_params(**kwargs) for steps_idx in self.steps]


    def fit(self, Xs, y=None):
        r"""
        Fit the pipeline to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        """
        for X_idx,X in enumerate(Xs):
            if self.same_pipeline_:
                self.pipeline_list_.append(deepcopy(make_pipeline(*self.steps, memory = self.memory, verbose = self.verbose).set_params(**self.kwargs)))
            self.pipeline_list_[X_idx].fit(X, y)
        return self

    def transform(self, Xs):
        r"""
        Transform the input data by applying the pipelines.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples_i, n_features_i)
        """

        transformed_Xs = [self.pipeline_list_[X_idx].transform(X) for X_idx, X in enumerate(Xs)]
        return transformed_Xs


    def predict(self, Xs):
        r"""
        Predict samples by using the fitted pipelines.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        labels : list of array-likes, shape (n_samples,)
            The predicted data.
        """
        labels = [self.pipeline_list_[X_idx].predict(X) for X_idx, X in enumerate(Xs)]
        return labels


    def fit_predict(self, Xs):
        r"""
        Fit the pipeline to the input data and predict samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        labels : list of array-likes, shape (n_samples,)
            The predicted data.
        """

        labels = self.fit(Xs).predict(Xs)
        return labels

