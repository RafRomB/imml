from typing import Union

from . import check_Xs


def convert_dataset_format(Xs: list, to_list: bool = False, to_dict: bool = True,
                           keys: list = None) -> Union[list, dict]:
    r"""
    Convert the format of a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    to_list : bool, default=False
        Convert from dict to list.
    to_dict : bool, default=False
        Convert from list to dict.
    keys : list, default=None
        keys for the dict. If None, it will use numbers starting from 0. Only used when to_dict is True.

    Returns
    -------
    transformed_Xs: dict of array-likes.
        - Xs length: n_mods
        - Xs[key] shape: (n_samples, n_features_i)

    Examples
    --------
    >>> from imml.utils.convert_dataset_format import convert_dataset_format    >>> import numpy as np
    >>> import pandas as pd
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> convert_dataset_format(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    if to_list:
        transformed_Xs = list(Xs.values())
    elif to_dict:
        if keys is None:
            keys = list(range(len(Xs)))
        transformed_Xs = {key:X for key,X in zip(keys, Xs)}
    else:
        transformed_Xs = Xs

    return transformed_Xs
