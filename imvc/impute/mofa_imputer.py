import numpy as np
import pandas as pd

from ..decomposition import MOFA
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
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.impute import MOFAImputer
    >>> from imvc.ampute import Amputer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> amp = Amputer(p=0.3, random_state=42)
    >>> Xs = amp.fit_transform(Xs)
    >>> transformer = MOFAImputer()
    >>> transformer.fit_transform(Xs)
    """

    def transform(self, Xs : list):
        r"""
        Transform the input data by filling missing samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features_i)
            The transformed data with filled missing samples.
        """

        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        ws = self.weights_
        winv = [np.linalg.pinv(w) for w in ws]
        transformed_Xs = [np.dot(X, w.T) for X,w in zip(Xs, winv)]
        transformed_Xs = self._impute(Xs=Xs, transformed_Xs=transformed_Xs, weights=ws)

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index) for X,transformed_X in zip(Xs,transformed_Xs)]
        return transformed_Xs
