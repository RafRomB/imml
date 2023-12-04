import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetUtils:
    r"""
    A utility class that provides general methods for working with multi-view datasets.
    """

    @staticmethod
    def add_random_noise_to_views(Xs: list, p, assess_percentage: bool = True, random_state: int = None, stratify = None):
        r"""
        Randomly drop samples in a multi-view dataset to convert it into an incomplete multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default=None
            If int, random_state is the seed used by the random number generator.
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
        >>> Xs = DatasetUtils.add_random_noise_to_views(Xs = Xs, p = [0.2, 0.5])
        """
        n_views = len(Xs)
        if not isinstance(p, list):
            p = [p]
        if len(p) != n_views:
            p = p*n_views

        if assess_percentage:
            p = [prob/len(p) for prob in p]
            sample_names = Xs[0].index
            total_len = len(sample_names)
            common_samples, _ = train_test_split(sample_names, train_size= round(1 - sum(p), 2),
                                       random_state=random_state, shuffle= True, stratify=stratify)
            sampled_names = copy.deepcopy(common_samples)

            if len(set(p)) == 1:
                n_unique_samples = total_len - len(common_samples)
                n_unique_samples_view= [n_unique_samples // n_views] * n_views
                n_unique_samples_view = np.full(n_views, n_unique_samples_view)
                n_unique_samples_view[:n_unique_samples % n_views] += 1
            else:
                n_unique_samples_view = [int(p_view*total_len) for p_view in p]

            imvd = []
            for X_idx,X in enumerate(Xs):
                x_per_view = X.drop(sampled_names).index
                if X_idx != n_views-1:
                    x_per_view, _ = train_test_split(x_per_view, train_size= n_unique_samples_view[X_idx],
                                                     random_state=random_state, shuffle=True,
                                                     stratify=stratify.loc[x_per_view] if stratify is not None else None)
                sampled_names = sampled_names.append(x_per_view)
                idxs_to_remove = common_samples.append(x_per_view)
                idxs_to_remove = X.index.difference(idxs_to_remove)
                X_ = copy.deepcopy(X)
                X_.loc[idxs_to_remove] = np.nan
                imvd.append(X_)
        else:
            imvd = []
            for X_idx,X in enumerate(Xs):
                idxs_to_remove = X.sample(frac = p[X_idx]/n_views,
                                          random_state = random_state + X_idx if random_state is not None else random_state).index
                X_ = copy.deepcopy(X)
                X_.loc[idxs_to_remove] = np.nan
                imvd.append(X_)
        return imvd


    @staticmethod
    def get_missing_view_panel(Xs: list):
        r"""
        Get the missing view panel of an incomplete multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        missing_view_panel: pd.DataFrame with binary values indicating missing views for each sample (1 has the view,
        0 otherwise).

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> missing_view_panel = DatasetUtils.get_missing_view_panel(Xs = Xs)
        """

        missing_view_panel = pd.concat([X.isna().all(1) for X in Xs], axis = 1)
        missing_view_panel = (~missing_view_panel).astype(int)
        return missing_view_panel


    @staticmethod
    def get_n_views(Xs: list):
        r"""
        Get the number of views of a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        n_views: number of views.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> missing_view_panel = DatasetUtils.get_n_views(Xs = Xs)
        """

        n_views = len(Xs)
        return n_views


    # @staticmethod
    # def get_sample_names(Xs: list):
    #     r"""
    #     Get all the samples in a multi-view dataset.
    #
    #     Parameters
    #     ----------
    #     Xs : list of array-likes
    #         - Xs length: n_views
    #         - Xs[i] shape: (n_samples_i, n_features_i)
    #         A list of different views.
    #
    #     Returns
    #     -------
    #     samples: pd.Index with all samples.
    #
    #     Examples
    #     --------
    #     >>> from imvc.utils import DatasetUtils
    #     >>> from imvc.datasets import LoadDataset
    #
    #     >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    #     >>> samples = DatasetUtils.get_sample_names(Xs = Xs)
    #     """
    #
    #     samples = pd.Index(set(sum([X.index.to_list() for X in Xs], [])))
    #     return samples


    @staticmethod
    def get_complete_sample_names(Xs: list):
        r"""
        Get complete samples in a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        samples: pd.Index with complete samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_complete_sample_names(Xs = Xs)
        """

        samples = DatasetUtils.get_missing_view_panel(Xs=Xs)
        samples = samples[samples.all(1)].index
        return samples


    @staticmethod
    def get_incomplete_sample_names(Xs: list):
        r"""
        Get incomplete samples in a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        samples: pd.Index with all samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_incomplete_sample_names(Xs = Xs)
        """

        samples = DatasetUtils.get_missing_view_panel(Xs=Xs)
        samples = samples[~samples.all(1)].index
        return samples


    # @staticmethod
    # def get_n_samples(Xs: list):
    #     r"""
    #     Get the number of samples in a multi-view dataset.
    #
    #     Parameters
    #     ----------
    #     Xs : list of array-likes
    #         - Xs length: n_views
    #         - Xs[i] shape: (n_samples_i, n_features_i)
    #         A list of different views.
    #
    #     Returns
    #     -------
    #     int: number of samples.
    #
    #     Examples
    #     --------
    #     >>> from imvc.utils import DatasetUtils
    #     >>> from imvc.datasets import LoadDataset
    #
    #     >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    #     >>> samples = DatasetUtils.get_n_samples(Xs = Xs)
    #     """
    #
    #     n_samples = len(DatasetUtils.get_sample_names(Xs=Xs))
    #     return n_samples


    @staticmethod
    def get_n_incomplete_samples(Xs: list):
        r"""
        Get the number of incomplete samples in a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        int: number of incomplete samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_n_incomplete_samples(Xs = Xs)
        """

        n_samples = len(DatasetUtils.get_incomplete_sample_names(Xs=Xs))
        return n_samples


    @staticmethod
    def get_n_complete_samples(Xs: list):
        r"""
        Get the number of complete samples in a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        int: number of complete samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_n_complete_samples(Xs = Xs)
        """

        n_samples = len(DatasetUtils.get_complete_sample_names(Xs=Xs))
        return n_samples


    @staticmethod
    def get_percentage_complete_samples(Xs: list):
        r"""
        Get the percentage of complete samples in a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        int: percentage of complete samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_percentage_complete_samples(Xs = Xs)
        """

        percentage_samples = DatasetUtils.get_n_complete_samples(Xs=Xs) / len(Xs[0]) * 100
        percentage_samples = int(percentage_samples)
        return percentage_samples


    @staticmethod
    def get_percentage_incomplete_samples(Xs: list):
        r"""
        Get the percentage of incomplete samples in a multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        int: percentage of incomplete samples.

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> samples = DatasetUtils.get_percentage_incomplete_samples(Xs = Xs)
        """

        percentage_samples = DatasetUtils.get_n_incomplete_samples(Xs=Xs) / len(Xs[0]) * 100
        percentage_samples = int(percentage_samples)
        return percentage_samples


    @staticmethod
    def shuffle_imvd(Xs: list, random_state: int = None):
        r"""
        Shuffle the dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
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


    # @staticmethod
    # def sort_y_based_on_imvd(Xs: list, y):
    #     r"""
    #     Order the target based on a multi-view dataset.
    #
    #     Parameters
    #     ----------
    #     Xs : list of array-likes
    #         - Xs length: n_views
    #         - Xs[i] shape: (n_samples_i, n_features_i)
    #         A list of different views.
    #
    #     Returns
    #     -------
    #     y : pd.Series
    #         Array with labels
    #
    #     Examples
    #     --------
    #     >>> from imvc.utils import DatasetUtils
    #     >>> from imvc.datasets import LoadDataset
    #     >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    #     >>> Xs = DatasetUtils.sort_y_based_on_imvd(Xs = Xs, y = y)
    #     """
    #
    #     y = y.loc[pd.concat([y.loc[X.index] for X in Xs]).index.drop_duplicates()]
    #     return y


    def remove_missing_sample_from_view(Xs: list):
        r"""
        Remove missing samples from each specific views.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        Xs: list of array-likes.
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
        """

        Xs = [X.loc[X.isna().all(1).index] for X in Xs]
        return Xs


    def force_all_samples(Xs: list):
        r"""
        Add missing samples to each view, in a way that all the views will have all samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        Xs: list of array-likes.
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        """

        samples = set(sum([X.index.to_list() for X in Xs], []))
        Xs = [pd.concat([X, pd.DataFrame(np.nan, index= pd.Index(samples).difference(X.index),
                                         columns= X.columns)]).loc[samples] for X in Xs]
        return Xs


    def convert_mvd_from_list_to_dict(Xs: list, keys: list):
        r"""
        Convert a multi-view dataset in list format to a dict format.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        Xs: dict of array-likes.
            - Xs length: n_views
            - Xs[key] shape: (n_samples, n_features_i)
        """

        Xs = {key:X for key,X in zip(keys, Xs)}
        return Xs


    def convert_mvd_from_dict_to_list(Xs: list, keys: list):
        r"""
        Convert a multi-view dataset in list format to a dict format.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        """

        Xs = list(Xs.values())
        return Xs




    def select_complete_samples(Xs: list):
        r"""
        Remove samples with missing views from the whole multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        Xs: list of array-likes.
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        """

        samples = DatasetUtils.get_complete_sample_names(Xs=Xs)
        Xs = [X.loc[samples] for X in Xs]
        return Xs


    def select_incomplete_samples(Xs: list):
        r"""
        Remove samples with no missing views from the whole multi-view dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        Xs: list of array-likes.
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        """

        samples = DatasetUtils.get_incomplete_sample_names(Xs=Xs)
        Xs = [X.loc[samples] for X in Xs]
        return Xs



