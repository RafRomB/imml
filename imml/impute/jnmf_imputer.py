import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from ..decomposition import JNMF


class JNMFImputer(JNMF):
    r"""
    Impute missing data in a dataset using the `JNMF` method.

    This class extends the `JNMF` class to provide functionality for filling in incomplete samples by
    addressing both block-wise and feature-wise missing data. As a subclass of `JNMF`, `JNMFImputer` inherits all
    input parameters and attributes from `JNMF`. Consequently, it uses the same `fit` method as `JNMF`
    training the model.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import JNMFImputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = JNMFImputer(n_components = 5)
    >>> labels = transformer.fit_transform(Xs)
    """


    def __init__(self, filling: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filling = filling


    def transform(self, Xs):
        r"""
        Impute unseen data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features_i)
            The transformed data with filled missing samples.
        """
        transformed_Xs = [np.dot(transformed_X + V, H.T)
                          for transformed_X,V,H in zip(super().transform(Xs), self.V_, self.H_)]

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        return transformed_Xs


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

        if self.filling:
            transformed_Xs_jnmf = [SimpleImputer().set_output(transform="pandas").fit_transform(X) for X in Xs]
            transformed_Xs_jnmf = super().fit_transform(transformed_Xs_jnmf)
        else:
            transformed_Xs_jnmf = super().fit_transform(Xs)
        transformed_Xs = []
        for X, V, H in zip(Xs, self.V_, self.H_):
            transformed_X = np.dot(transformed_Xs_jnmf + V, H.T)
            if isinstance(Xs[0], pd.DataFrame):
                transformed_X = X.fillna(pd.DataFrame(transformed_X, index=X.index, columns=X.columns))
            else:
                transformed_X = pd.DataFrame(X).fillna(pd.DataFrame(transformed_X))
            transformed_Xs.append(transformed_X)

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        elif self.transform_ == "numpy":
            transformed_Xs = [transformed_X.values for transformed_X in transformed_Xs]

        return transformed_Xs

