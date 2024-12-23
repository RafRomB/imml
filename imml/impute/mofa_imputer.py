import numpy as np
import pandas as pd

from ..decomposition.mofa import MOFA
from ..utils import check_Xs


class MOFAImputer(MOFA):
    r"""
    Impute missing data in a dataset using the `MOFA` method.

    This class extends the `MOFA` class to provide functionality for filling in incomplete samples by
    addressing both block-wise and feature-wise missing data. As a subclass of MOFA, `MOFAImputer` inherits all
    input parameters and attributes from `MOFA`. Consequently, it uses the same `fit` method as MOFA for
    training the model.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import MOFAImputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = MOFAImputer(n_components = 5)
    >>> labels = transformer.fit_transform(Xs)
    """


    def fit_transform(self, Xs, y = None, **fit_params):
        r"""
        Fit to data, then impute them.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different mods.
        y : Ignored
            Not used, present here for API consistency by convention.
        fit_params : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        transformed_X : array-likes of shape (n_samples, n_components)
            The transformed data with filled missing samples.
        """

        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        self.fit(Xs)
        transformed_Xs = []
        for X, w in zip(Xs, self.weights_):
            transformed_X = np.dot(np.nan_to_num(self.factors_, nan=0.0), w.T)
            transformed_X = pd.DataFrame(transformed_X, columns=X.columns)
            transformed_Xs.append(X.fillna(transformed_X))

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index) for X,transformed_X in zip(Xs,transformed_Xs)]
        elif self.transform_ == "numpy":
            transformed_Xs = [transformed_X.values for transformed_X in transformed_Xs]

        return transformed_Xs

