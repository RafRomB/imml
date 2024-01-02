import copy

import numpy as np
import pandas as pd
from pyampute import MultivariateAmputation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from imvc.utils import DatasetUtils


class Ampute(BaseEstimator, TransformerMixin):

    def __init__(self, p, mechanism: str = "ED", random_state: int = None,
                 assess_percentage: bool = True, stratify=None):
        r"""
        Generate view missingness patterns in complete multi-view datasets.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        mechanism: str, default="EDM"
            One of ["EDM", 'MCAR', 'MAR', 'MNAR'].
        random_state: int, default=None
            If int, random_state is the seed used by the random number generator.
        assess_percentage: bool
            If False, each view is dropped independently.
        stratify: array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels.

        Returns
        -------
        imvd : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

         Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0)
        >>> Xs = DatasetUtils.ampute(Xs = Xs, p = [0.2, 0.5])
        """
        possible_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR', 'MAR+MNAR']
        if mechanism not in possible_mechanisms:
            raise ValueError(f"Invalid mechanism. Expected one of: {possible_mechanisms}")

        self.p = p
        self.mechanism = mechanism
        self.random_state = random_state
        self.assess_percentage = assess_percentage
        self.stratify = stratify


    def fit(self, Xs: list, y=None):
        n_views = len(Xs)
        self.n_views = n_views
        if not isinstance(self.p, list):
            self.p = [self.p]
        if len(self.p) != n_views:
            self.p *= n_views
        return self


    def transform(self, Xs: list, y = None):

        if self.mechanism == "EDM":
            if self.assess_percentage:
                p = [prob / len(self.p) for prob in self.p]
                sample_names = Xs[0].index
                total_len = len(sample_names)
                common_samples, _ = train_test_split(sample_names, train_size=round(1 - sum(p), 2),
                                                     random_state=self.random_state, shuffle=True, stratify=self.stratify)
                sampled_names = copy.deepcopy(common_samples)

                if len(set(p)) == 1:
                    n_unique_samples = total_len - len(common_samples)
                    n_unique_samples_view = [n_unique_samples // self.n_views] * self.n_views
                    n_unique_samples_view = np.full(self.n_views, n_unique_samples_view)
                    n_unique_samples_view[:n_unique_samples % self.n_views] += 1
                else:
                    n_unique_samples_view = [int(p_view * total_len) for p_view in p]

                transformed_Xs = []
                for X_idx, X in enumerate(Xs):
                    x_per_view = X.drop(sampled_names).index
                    if X_idx != self.n_views - 1:
                        x_per_view, _ = train_test_split(x_per_view, train_size=n_unique_samples_view[X_idx],
                                                         random_state=self.random_state, shuffle=True,
                                                         stratify=self.stratify.loc[
                                                             x_per_view] if self.stratify is not None else None)
                    sampled_names = sampled_names.append(x_per_view)
                    idxs_to_remove = common_samples.append(x_per_view)
                    idxs_to_remove = X.index.difference(idxs_to_remove)
                    X_ = copy.deepcopy(X)
                    X_.loc[idxs_to_remove] = np.nan
                    transformed_Xs.append(X_)
            else:
                transformed_Xs = []
                for X_idx, X in enumerate(Xs):
                    idxs_to_remove = X.sample(frac=p[X_idx] / self.n_views,
                                              random_state=self.random_state + X_idx if self.random_state is not None else self.random_state).index
                    X_ = copy.deepcopy(X)
                    X_.loc[idxs_to_remove] = np.nan
                    transformed_Xs.append(X_)

        else:
            pseudo_missing_view_profile = np.random.default_rng(seed=self.random_state).standard_normal((len(Xs[0]), len(Xs)))
            pseudo_missing_view_profile = pd.DataFrame(pseudo_missing_view_profile)

            if pseudo_missing_view_profile.shape[1] > 2:
                n_views_to_remove = round(pseudo_missing_view_profile.shape[1] * 0.5 +
                                          pd.Series([0.1, -0.1]).sample(1, random_state=self.random_state).iloc[0])
            else:
                n_views_to_remove = 1
            views_to_remove = pseudo_missing_view_profile.columns.to_series().sample(n=n_views_to_remove,
                                                                            random_state=self.random_state)
            views_to_remove = pseudo_missing_view_profile.columns[views_to_remove]

            amp = MultivariateAmputation(patterns=[{"incomplete_vars": views_to_remove, "mechanism": self.mechanism}],
                                         seed= self.random_state)
            pseudo_missing_view_profile = amp.fit_transform(pseudo_missing_view_profile).fillna(0).astype(int)
            pseudo_missing_view_profile[pseudo_missing_view_profile.notnull()] = 1
            pseudo_missing_view_profile = pseudo_missing_view_profile.fillna(0).astype(int)
            transformed_Xs = DatasetUtils.convert_mvd_in_imvd(Xs=Xs, missing_view_profile=pseudo_missing_view_profile)

        return transformed_Xs


    # def fit_transform(self, Xs, y=None):
    #     transformed_Xs = self.fit(Xs=Xs, y=y).transform(Xs=Xs, y=y)
    #     return transformed_Xs


