import numpy as np
import pandas as pd
from sklearn.utils._random import sample_without_replacement


class DatasetUtils:
    r"""
    A utility class that provides general methods for working with incomplete multi-view datasets.
    """


    def create_imvd_from_mvd(self, Xs: list, p, random_state: int = None):
        r"""
        Creates a random panel for transforming a complete multi-view dataset into an incomplete one by randomly
        removing samples from each view.
        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        p: list or int
            The percentaje that each view will have for missing samples. If p is int, all the views will have the
            same percentaje.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        imvd : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

         Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import load_incomplete_nutrimouse
        >>> Xs = load_incomplete_nutrimouse(p = [0])
        >>> DatasetUtils().create_imvd_from_mvd(Xs = Xs, p = [0.2, 0.5])
        """

        sample_view_panel = self.create_random_missing_views(Xs = Xs, p = p, random_state = random_state)
        imvd = self.add_missing_views(Xs= Xs, sample_view_panel=sample_view_panel)
        return imvd


    @staticmethod
    def create_random_missing_views(Xs: list, p, random_state: int = None):
        r"""
        Creates a random panel for transforming a complete multi-view dataset into an incomplete one by randomly
        removing samples from each view.
        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        p: list or int
            The percentaje that each view will have for missing samples. If p is int, all the views will have the
            same percentaje.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.
.

        Returns
        -------
        sample_view_panel: A DataFrame with binary values indicating missing views for each
        sample (1 indicates the view is present, 0 indicates it is missing).

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import load_incomplete_nutrimouse
        >>> Xs = load_incomplete_nutrimouse(p = [0])
        >>> DatasetUtils.create_random_missing_views(Xs = Xs, p = [0.2, 0.5])
        """
        if not isinstance(p, list):
            p = [p]
        if len(p) != len(Xs):
            p = p*len(Xs)

        n_samples = len(Xs[0])
        sample_view_panel = []
        for X_idx in range(len(Xs)):
            sample_view = np.array([1] * n_samples)
            if random_state is not None:
                random_state += 1
            missing = sample_without_replacement(n_population = n_samples, n_samples = int(p[X_idx] * n_samples),
                                                 random_state = random_state)
            sample_view[missing] = 0
            sample_view_panel.append(sample_view.tolist())
        sample_view_panel = pd.DataFrame(sample_view_panel, columns = Xs[X_idx].index).transpose()
        sample_view_panel[sample_view_panel.sum(1) == 0] = 1
        return sample_view_panel


    @staticmethod
    def add_missing_views(Xs: list, sample_view_panel: pd.DataFrame):
        r"""
        Transform a complete multi-view dataset in an incomplete multi-view problem.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        sample_view_panel: pd.DataFrame with binary values indicating missing views for each sample (1 has the view,
        0 otherwise).

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import load_incomplete_nutrimouse
        >>> Xs = load_incomplete_nutrimouse(p = [0])
        >>> sample_view_panel = DatasetUtils.create_random_missing_views(Xs = Xs, p = [0.2, 0.5])
        >>> DatasetUtils.add_missing_views(Xs = Xs, sample_view_panel = sample_view_panel)
        """

        imvd = Xs.copy()
        missing_views = sample_view_panel == 1
        for X_idx, X in enumerate(Xs):
            imvd[X_idx] = X[missing_views.loc[:, X_idx]]
        return imvd


    @staticmethod
    def get_missing_view_panel(Xs: list):
        r"""
        Get the missing view panel of an incomplete multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        sample_view_panel: pd.DataFrame with binary values indicating missing views for each sample (1 has the view,
        0 otherwise).

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import load_incomplete_nutrimouse
        >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
        >>> DatasetUtils.get_missing_view_panel(Xs = Xs)
        """

        sample_view_panel = pd.concat([X.index.to_series() for X in Xs], axis = 1).sort_index()
        sample_view_panel = sample_view_panel.mask(sample_view_panel.isna(), 0).where(sample_view_panel.isna(), 1).astype(int)
        return sample_view_panel






