from sklearn.preprocessing import FunctionTransformer

from ..utils import check_Xs, DatasetUtils


class SelectCompleteSamples(FunctionTransformer):
    r"""
    Fill missing samples in different views of a dataset using a specified method.

    Parameters
    ----------
    value : str, optional (default='mean')
        The method to use for filling missing samples. Possible values:
        - 'mean': replace missing samples with the mean of each feature in the corresponding view
        - 'zeros': replace missing samples with zeros
        - 'nan': replace missing samples with NaN

    Attributes
    ----------
    features_view_mean_list_ : array-like of shape (n_views,)
        The mean value of each feature in the corresponding view, if value='mean'

    Examples
    --------
    >>> from imvc.datasets import LoadDataset

    >>> from imvc.transformers import FillMissingViews
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> transformer = FillMissingViews()
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self):
        super().__init__(select_complete_samples)


def select_complete_samples(Xs:list):
    r"""
    Transform the input data by filling missing samples.

    Parameters
    ----------
    Xs : list of array-likes
        - Xs length: n_views
        - Xs[i] shape: (n_samples_i, n_features_i)
        A list of different views.

    Returns
    -------
    transformed_Xs : list of array-likes, shape (n_samples, n_features)
        The transformed data with filled missing samples.
    """

    Xs = check_Xs(Xs, allow_incomplete=True, force_all_finite='allow-nan')
    sample_views = DatasetUtils.get_missing_view_profile(Xs=Xs)
    complete_samples = sample_views.all(axis= 1)
    transformed_Xs = [X[complete_samples] for X in Xs]
    return transformed_Xs


