import copy
from typing import Union
import numpy as np
import pandas as pd

from . import check_Xs
from ..impute import get_observed_mod_indicator, get_missing_mod_indicator


class DatasetUtils:
    r"""
    A utility class that provides general methods for working with multi-modal datasets.
    """

    @staticmethod
    def convert_to_imvd(Xs: list, observed_mod_indicator) -> list:
        r"""
        Generate block-wise missingness patterns in complete multi-modal datasets.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        observed_mod_indicator: array-like of shape (n_samples, n_mods)
            Boolean array-like indicating observed modalities for each sample.

        Returns
        -------
        transformed_Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.pipeline import make_pipeline
        >>> from imml.impute import ObservedModIndicator
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> transformer = make_pipeline(Amputer(p=0.2, mechanism="mcar", random_state=42),
                                        ObservedModIndicator().set_output(transformed="pandas"))
        >>> observed_mod_indicator = transformer.fit_transform(Xs)
        >>> DatasetUtils.convert_to_imvd(Xs = Xs, observed_mod_indicator = observed_mod_indicator)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        transformed_Xs = []
        if isinstance(observed_mod_indicator, pd.DataFrame):
            observed_mod_indicator = observed_mod_indicator.values
        for X_idx, X in enumerate(Xs):
            idxs_to_remove = observed_mod_indicator[:,X_idx] == False
            if isinstance(X, pd.DataFrame):
                X = X.values
            transformed_X = copy.deepcopy(X).astype(float)
            transformed_X[idxs_to_remove, :] = np.nan
            transformed_Xs.append(transformed_X)
        if isinstance(Xs[0], pd.DataFrame):
            transformed_Xs = [pd.DataFrame(transformed_X, columns=X.columns, index=X.index) for X, transformed_X in zip(Xs, transformed_Xs)]

        return transformed_Xs


    @staticmethod
    def get_summary(Xs: list) -> int:
        r"""
        Get a summary of an incomplete multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        summary: dict
            Summary of an incomplete multi-modal dataset.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> DatasetUtils.get_n_mods(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        summary = {
            "Complete samples": DatasetUtils.get_n_complete_samples(Xs),
            "Incomplete samples": DatasetUtils.get_n_incomplete_samples(Xs),
            "Observed samples per modality": [len(Xs[0]) - len(X_id) for X_id in
                                              DatasetUtils.get_missing_samples_by_mod(Xs)],
            "Missing samples per modality": [len(X_id) for X_id in
                                             DatasetUtils.get_missing_samples_by_mod(Xs)],
            "% Observed samples per modality": [round((len(Xs[0]) - len(X_id)) / len(Xs[0]) * 100) for X_id in
                                                DatasetUtils.get_missing_samples_by_mod(Xs)],
            "% Missing samples per modality": [round(len(X_id) / len(Xs[0]) * 100) for X_id in
                                               DatasetUtils.get_missing_samples_by_mod(Xs)],
        }
        return summary


    @staticmethod
    def get_n_mods(Xs: list) -> int:
        r"""
        Get the number of modalities of a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        n_mods: int
            Number of modalities.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> DatasetUtils.get_n_mods(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        n_mods = len(Xs)
        return n_mods


    @staticmethod
    def get_n_samples_by_mod(Xs: list) -> int:
        r"""
        Get the number of samples in each modality.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        n_samples_by_mod: pd.Series
            Number of samples in each modality.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> DatasetUtils.get_n_samples_by_mod(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        n_samples_by_mod = get_observed_mod_indicator(Xs)
        n_samples_by_mod = n_samples_by_mod.sum(axis=0)
        return n_samples_by_mod


    @staticmethod
    def get_complete_sample_names(Xs: list) -> pd.Index:
        r"""
        Get complete samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        samples: pd.Index
            Sample names with full data.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_complete_sample_names(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        samples = get_observed_mod_indicator(Xs)
        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame(samples)
        samples = samples[samples.all(1)].index
        return samples


    @staticmethod
    def get_incomplete_sample_names(Xs: list) -> pd.Index:
        r"""
        Get incomplete samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        samples: pd.Index
            Sample names with incomplete data.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_incomplete_sample_names(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        samples = get_observed_mod_indicator(Xs)
        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame(samples)
        samples = samples[~samples.all(1)].index
        return samples


    @staticmethod
    def get_sample_names(Xs: list) -> pd.Index:
        r"""
        Get samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different modalities.

        Returns
        -------
        samples: pd.Index (n_samples,)
            Sample names.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_sample_names(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]
        samples = [X.index.to_list() for X in Xs]
        samples = [x for xs in samples for x in xs]
        samples = pd.Index(sorted(set(samples), key=samples.index))
        return samples


    @staticmethod
    def get_samples_by_mod(Xs: list, return_as_list: bool = True) -> Union[list, dict]:
        r"""
        Get the samples for each modality in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        return_as_list : bool, default=True
            If True, the function will return a list; a dict otherwise.

        Returns
        -------
        samples: list or dict of pd.Index
            If list, each element in the list is the sample names for each modality. If dict, keys are the modalities and the
            values are the sample names.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_samples_by_mod(Xs = Xs)
        """
        observed_mod_indicator = get_observed_mod_indicator(Xs)
        if isinstance(Xs[0], pd.DataFrame):
            observed_mod_indicator = pd.DataFrame(observed_mod_indicator, index=Xs[0].index)
        else:
            observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
        if return_as_list:
            samples = [mod_profile[mod_profile].index for X_idx, mod_profile in observed_mod_indicator.items()]
        else:
            samples = {X_idx: mod_profile[mod_profile].index for X_idx, mod_profile in observed_mod_indicator.items()}
        return samples


    @staticmethod
    def get_missing_samples_by_mod(Xs: list, return_as_list: bool = True) -> Union[list, dict]:
        r"""
        Get the samples not present in each modality in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        return_as_list : bool, default=True
            If list, each element in the list is the sample names for each modality. If dict, keys are the modalities and the
            values are the sample names.

        Returns
        -------
        samples: dict of pd.Index or list of pd.Index.
            Dictionary or list of missing samples for each modality.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_missing_samples_by_mod(Xs = Xs)
        """

        observed_mod_indicator = get_observed_mod_indicator(Xs)
        if isinstance(Xs[0], pd.DataFrame):
            observed_mod_indicator = pd.DataFrame(observed_mod_indicator, index=Xs[0].index)
        else:
            observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
        if return_as_list:
            samples = [mod_profile[mod_profile == False].index.to_list()
                       for X_idx, mod_profile in observed_mod_indicator.items()]
        else:
            samples = {X_idx: mod_profile[mod_profile == False].index.to_list()
                       for X_idx, mod_profile in observed_mod_indicator.items()}
        return samples


    @staticmethod
    def get_n_complete_samples(Xs: list) -> int:
        r"""
        Get the number of complete samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        n_samples: int
            number of complete samples.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_n_complete_samples(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        n_samples = len(DatasetUtils.get_complete_sample_names(Xs=Xs))
        return n_samples


    @staticmethod
    def get_n_incomplete_samples(Xs: list) -> int:
        r"""
        Get the number of incomplete samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        n_samples: int
            number of incomplete samples.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_n_incomplete_samples(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        n_samples = len(DatasetUtils.get_incomplete_sample_names(Xs=Xs))
        return n_samples


    @staticmethod
    def get_percentage_complete_samples(Xs: list) -> float:
        r"""
        Get the percentage of complete samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        percentage_samples: float
            percentage of complete samples.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_percentage_complete_samples(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        percentage_samples = DatasetUtils.get_n_complete_samples(Xs=Xs) / len(Xs[0]) * 100
        return percentage_samples


    @staticmethod
    def get_percentage_incomplete_samples(Xs: list) -> float:
        r"""
        Get the percentage of incomplete samples in a multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        percentage_samples: float
            percentage of incomplete samples.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.get_percentage_incomplete_samples(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        percentage_samples = DatasetUtils.get_n_incomplete_samples(Xs=Xs) / len(Xs[0]) * 100
        return percentage_samples


    @staticmethod
    def remove_missing_sample_from_mod(Xs: list) -> list:
        r"""
        Remove missing samples from each specific modality.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_Xs: list of array-likes.
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different modalities.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> from imml.ampute import Amputer
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
        >>> DatasetUtils.remove_missing_sample_from_mod(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        observed_mod_indicator = get_missing_mod_indicator(Xs)
        if observed_mod_indicator.any().any():
            pandas_format = isinstance(Xs[0], pd.DataFrame)
            if pandas_format:
                cols = [X.columns for X in Xs]
                samples = Xs[0].index
                Xs = [X.values for X in Xs]
            masks = [np.invert(np.isnan(X).all(1)) for X in Xs]
            transformed_Xs = [X[mask] for X, mask in zip(Xs, masks)]
            if pandas_format:
                transformed_Xs = [pd.DataFrame(transformed_X, index=samples[mask], columns=col)
                                  for transformed_X, mask, col in zip(transformed_Xs, masks, cols)]
        else:
            transformed_Xs = Xs
        return transformed_Xs


    @staticmethod
    def convert_mmd_from_list_to_dict(Xs: list, keys: list = None) -> dict:
        r"""
        Convert a multi-modal dataset in list format to a dict format.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        keys : list, default=None
            keys for the dict. If None, it will use numbers starting from 0.

        Returns
        -------
        transformed_Xs: dict of array-likes.
            - Xs length: n_mods
            - Xs[key] shape: (n_samples, n_features_i)

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> DatasetUtils.convert_mmd_from_list_to_dict(Xs = Xs)
        """
        Xs = check_Xs(Xs=Xs, force_all_finite="allow-nan")
        if keys is None:
            keys = list(range(len(Xs)))

        transformed_Xs = {key:X for key,X in zip(keys, Xs)}
        return transformed_Xs


    @staticmethod
    def convert_mmd_from_dict_to_list(Xs: dict) -> list:
        r"""
        Convert a multi-modal dataset in list format to a dict format.

        Parameters
        ----------
        Xs : dict of array-likes
            - Xs length: n_mods
            - Xs[key] shape: (n_samples, n_features_i)

        Returns
        -------
        transformed_Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imml.utils import DatasetUtils
        >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
        >>> Xs_dict = DatasetUtils.convert_mmd_from_list_to_dict(Xs = Xs)
        >>> DatasetUtils.convert_mmd_from_dict_to_list(Xs = Xs_dict)
        """
        transformed_Xs = list(Xs.values())
        return transformed_Xs



