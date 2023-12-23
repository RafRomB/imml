import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetUtils:
    r"""
    A utility class that provides general methods for working with multi-view datasets.
    """

    @staticmethod
    def convert_mvd_in_imvd(Xs: list, missing_view_profile = pd.DataFrame):
        r"""
        Generate view missingness patterns in complete multi-view datasets.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        missing_view_profile: pd.DataFrame
            pd.DataFrame with binary values indicating missing views for each sample (1 has the view, 0 otherwise).

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
        transformed_Xs = []
        for X_idx, X in enumerate(Xs):
            idxs_to_remove = missing_view_profile[missing_view_profile[X_idx] == 0].index
            transformed_X = copy.deepcopy(X)
            transformed_X.loc[idxs_to_remove] = np.nan
            transformed_Xs.append(transformed_X)

        return transformed_Xs


    @staticmethod
    def get_missing_view_profile(Xs: list):
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
        missing_view_profile: pd.DataFrame with binary values indicating missing views for each sample (1 has the view,
        0 otherwise).

        Examples
        --------
        >>> from imvc.utils import DatasetUtils
        >>> from imvc.datasets import LoadDataset

        >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
        >>> missing_view_profile = DatasetUtils.get_missing_view_profile(Xs = Xs)
        """

        missing_view_profile = pd.concat([X.isna().all(1) for X in Xs], axis = 1)
        missing_view_profile = (~missing_view_profile).astype(int)
        return missing_view_profile


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
        >>> missing_view_profile = DatasetUtils.get_n_views(Xs = Xs)
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

        samples = DatasetUtils.get_missing_view_profile(Xs=Xs)
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

        samples = DatasetUtils.get_missing_view_profile(Xs=Xs)
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

        missing_view_profile = DatasetUtils.get_missing_view_profile(Xs)
        samples = missing_view_profile.sample(frac = 1., random_state = random_state).index
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



