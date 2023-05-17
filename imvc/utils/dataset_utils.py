import numpy as np
import pandas as pd


class DatasetUtils:
    r"""
    A utility class that provides general methods for working with incomplete multi-view datasets.
    """

    @staticmethod
    def convert_mvd_into_imvd(Xs: list, p, random_state: int = None):
        r"""
        Randomly drop samples in a multi-view dataset to convert it into an incomplete multi-view dataset.

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
        >>> from utils import DatasetUtils
        >>> from datasets import LoadDataset
        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0)
        >>> Xs = DatasetUtils.convert_mvd_into_imvd(Xs = Xs, p = [0.2, 0.5])
        """
        if not isinstance(p, list):
            p = [p]
        if len(p) != len(Xs):
            p = p*len(Xs)

        imvd = [X.drop(X.sample(frac = p[X_idx],
                                random_state = random_state + X_idx if random_state is not None else random_state).index)
                for X_idx,X in enumerate(Xs)]
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
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> missing_view_panel = DatasetUtils.get_missing_view_panel(Xs = Xs)
        """

        sample_view_panel = pd.concat([X.index.to_series() for X in Xs], axis = 1).sort_index()
        sample_view_panel = sample_view_panel.mask(sample_view_panel.isna(), 0).where(sample_view_panel.isna(), 1).astype(int)
        return sample_view_panel


    @staticmethod
    def get_sample_names(Xs: list):
        r"""
        Get all the samples in an incomplete multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        samples: pd.Index with all samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_sample_names(Xs = Xs)
        """

        samples = pd.Index(set(sum([X.index.to_list() for X in Xs], [])))
        return samples


    @staticmethod
    def shuffle_imvd(Xs: list, random_state: int = None):
        r"""
        Shuffle the dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs: list of array-likes.
            Incomplete multi-view dataset with shuffled samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> Xs = DatasetUtils.shuffle_imvd(Xs = Xs)
        """

        missing_view_panel = DatasetUtils.get_missing_view_panel(Xs)
        samples = missing_view_panel.sample(frac = 1., random_state = random_state).index
        Xs = [X.loc[samples.intersection(X.index)] for X in Xs]
        return Xs







