import numpy as np
import pandas as pd

from ..utils import check_Xs, DatasetUtils


def get_summary(Xs: list, modalities: list = None, one_row: bool = False, compute_pct: bool = True) -> dict:
    r"""
    Get a summary of an incomplete multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    modalities : list, default=None
        Name of each modality. By default, it will be set to the modality index. Only applicable when one_row is False.
    one_row : bool, default=False
        If True, return a one-row summary of the dataset. If False, each row will correspond to a modality.
    compute_pct : bool, default=True
        If True, compute percent of each value.

    Returns
    -------
    summary: dict
        Summary of an incomplete multi-modal dataset.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.statistics import get_summary
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> get_summary(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    n_samples = len(Xs[0])
    if one_row:
        summary = {
            "Complete samples": DatasetUtils.get_n_complete_samples(Xs),
            "Incomplete samples": DatasetUtils.get_n_incomplete_samples(Xs),
            "Observed samples per modality": [n_samples - len(X_id) for X_id in
                                              DatasetUtils.get_missing_samples_by_mod(Xs)],
            "Missing samples per modality": [len(X_id) for X_id in
                                             DatasetUtils.get_missing_samples_by_mod(Xs)],
            "% Observed samples per modality": [round((n_samples - len(X_id)) / n_samples * 100) for X_id in
                                                DatasetUtils.get_missing_samples_by_mod(Xs)],
            "% Missing samples per modality": [round(len(X_id) / n_samples * 100) for X_id in
                                               DatasetUtils.get_missing_samples_by_mod(Xs)],
        }
        if compute_pct:
            summary = {
                **summary,
                "% Observed samples per modality": [round((n_samples - len(X_id)) / n_samples * 100) for X_id in
                                                    DatasetUtils.get_missing_samples_by_mod(Xs)],
                "% Missing samples per modality": [round(len(X_id) / n_samples * 100) for X_id in
                                                   DatasetUtils.get_missing_samples_by_mod(Xs)],
            }


    else:
        if modalities is None:
            modalities = list(range(len(Xs)))
        c_samples, m_samples, i_samples = [], [], []
        summary = {}
        for X, mod in zip(Xs, modalities):
            mod_c_samples = pd.DataFrame(X)[np.isfinite(X).all(axis=1)]
            mod_m_samples = pd.DataFrame(X)[np.isnan(X).all(axis=1)]
            mod_i_samples = pd.DataFrame(X)[np.isnan(X).any(axis=1) & np.isfinite(X).any(axis=1)]
            summary[mod] = {
                "Complete samples": len(mod_c_samples),
                "Missing samples": len(mod_m_samples),
                "Incomplete samples": len(mod_i_samples),
            }
            c_samples.append(mod_c_samples.index.to_series())
            m_samples.append(mod_m_samples.index.to_series())
            i_samples.append(mod_i_samples.index.to_series())
        summary["Total"] = {
            "Complete samples": (pd.concat(c_samples).value_counts() == len(Xs)).sum(),
            "Missing samples": (pd.concat(m_samples).value_counts() > 0).sum(),
            "Incomplete samples": (pd.concat(i_samples).value_counts() > 0).sum(),
        }
        if compute_pct:
            for mod in summary.keys():
                for k in list(summary[mod].keys()):
                    summary[mod][f"% {k}"] = summary[mod][k] / n_samples * 100
    return summary
