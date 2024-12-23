import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs


class SelectCompleteSamples(FunctionTransformer):
    r"""
    Remove incomplete samples from a multi-modal dataset. Apply `FunctionTransformer` (from `Scikit-learn`)
    with `select_complete_samples` as a function.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import SelectCompleteSamples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> transformer = SelectCompleteSamples()
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self):
        super().__init__(select_complete_samples)


def select_complete_samples(Xs: list):
    r"""
    Remove incomplete samples from a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features)

        A list of different mods.

    Returns
    -------
    transformed_Xs : list of array-likes, shape (n_samples, n_features_i)
        The transformed data.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import select_complete_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> select_complete_samples(Xs)
    """

    Xs = check_Xs(Xs, force_all_finite='allow-nan')
    pandas_format = isinstance(Xs[0], pd.DataFrame)
    if not pandas_format:
        Xs = [pd.DataFrame(X) for X in Xs]
    sample_modalitys = get_observed_mod_indicator(Xs)
    complete_samples = sample_modalitys.all(axis= 1)
    transformed_Xs = [X.loc[complete_samples] for X in Xs]
    if not pandas_format:
        transformed_Xs = [X.values for X in transformed_Xs]
    return transformed_Xs


