from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin


class MultiViewPipeline(BaseEstimator, TransformerMixin):
    r"""
    An estimator that applies the same pipeline to multiple views of data.

    Parameters
    ----------
    pipeline : scikit-learn pipeline object or list of scikit-learn pipeline object
        A scikit-learn pipeline object that will be used to estimate each view of data. If a list is provided,
        each pipeline will be applied on each view, otherwise the same pipeline will be applied on each view.

    Attributes
    ----------
    pipeline_list_ : list of pipeline (n_views,)
        A list of pipelines, one for each view of data.
    same_pipeline_ : boolean
        A booleaing indicating if the same pipeline will be applied on each view of data.

    Examples
    --------
    >>> from datasets import load_incomplete_nutrimouse
    >>> from pipelines import MultiViewPipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.pipeline import make_pipeline
    >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
    >>> pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3))
    >>> mv_pipeline = MultiViewPipeline(pipeline = pipeline)
    >>> mv_pipeline.fit_predict(Xs)
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.same_pipeline_ = False if isinstance(pipeline, list) else True
        self.pipeline_list_ = [] if self.same_pipeline_ else pipeline


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
                self.pipeline_list_.append(deepcopy(self.pipeline))
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

