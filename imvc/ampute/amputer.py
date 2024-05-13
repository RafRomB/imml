import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from scipy import optimize

from imvc.utils import DatasetUtils


class Amputer(BaseEstimator, TransformerMixin):
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

    Attributes
    ----------
    views_to_remove_ : array-like of shape (n_views * 0.5,)
        Views that will be removed from samples. Only available if mechanism != 'EDM'.

    Returns
    -------
    imvd : list of array-likes
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        A list of different views.

     Examples
    --------
    >>> from imvc.utils import DatasetUtils
    >>> from imvc.datasets import LoadDataset
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0)
    >>> Xs = DatasetUtils.ampute(Xs = Xs, p = [0.2, 0.5])
    """

    def __init__(self, p, mechanism: str = "EDM", random_state: int = None,
                 assess_percentage: bool = True, stratify=None,
                 opt:str = "logistic", p_obs:float = 0.1, q= 0.3, exclude_inputs:bool = True,
                 p_params= 0.3, cut='both', mcar=False):
        possible_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR', "PM"]
        if mechanism not in possible_mechanisms:
            raise ValueError(f"Invalid mechanism. Expected one of: {possible_mechanisms}")

        self.p = p
        self.mechanism = mechanism
        self.random_state = random_state
        self.assess_percentage = assess_percentage
        self.stratify = stratify
        self.opt = opt
        self.p_obs = p_obs
        self.q = q
        self.exclude_inputs = exclude_inputs
        self.p_params = p_params
        self.cut = cut
        self.mcar = mcar


    def fit(self, Xs: list, y=None):
        n_views = len(Xs)
        self.n_views = n_views

        if self.mechanism == "EDM":
            if not isinstance(self.p, list):
                self.p = [self.p]
            if len(self.p) != n_views:
                self.p *= n_views

        return self


    def transform(self, Xs: list, y=None):

        if self.mechanism == "EDM":
            if self.assess_percentage:
                p = [prob / len(self.p) for prob in self.p]
                sample_names = Xs[0].index
                total_len = len(sample_names)
                common_samples, _ = train_test_split(sample_names, train_size=round(1 - sum(p), 2),
                                                     random_state=self.random_state, shuffle=True,
                                                     stratify=self.stratify)
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
                    idxs_to_remove = X.sample(frac=self.p[X_idx] / self.n_views,
                                              random_state=self.random_state + X_idx if self.random_state is not None else self.random_state).index
                    X_ = copy.deepcopy(X)
                    X_.loc[idxs_to_remove] = np.nan
                    transformed_Xs.append(X_)

        else:
            pseudo_observed_view_indicator = np.random.default_rng(self.random_state).normal(size=(len(Xs[0]), self.n_views))
            pseudo_observed_view_indicator = self._produce_missing(X= pseudo_observed_view_indicator)
            pseudo_observed_view_indicator = pd.DataFrame(pseudo_observed_view_indicator, index=Xs[0].index)
            pseudo_observed_view_indicator[pseudo_observed_view_indicator.notnull()] = 1
            pseudo_observed_view_indicator = pseudo_observed_view_indicator.fillna(0).astype(int)
            transformed_Xs = DatasetUtils.convert_to_imvd(Xs=Xs, observed_view_indicator=pseudo_observed_view_indicator)

        return transformed_Xs


    def _produce_missing(self, X):
        """
        Generate missing values for specifics missing-data mechanism and proportion of missing values.

        Parameters
        ----------
        X : torch.DoubleTensor or np.ndarray, shape (n, d)
            Data for which missing values will be simulated.
            If a numpy array is provided, it will be converted to a pytorch tensor.
        p_miss : float
            Proportion of missing values to generate for variables which will have missing values.
        mecha : str,
                Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
        opt: str,
             For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
        p_obs : float
                If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
        q : float
            If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

        Returns
        ----------
        A dictionnary containing:
        'X_init': the initial data matrix.
        'X_incomp': the data with the generated missing values.
        'mask': a matrix indexing the generated missing values.s
        """

        if self.mechanism == "MAR":
            mask = self._MAR_mask(X=X, p=self.p, p_obs=self.p_obs)
        elif self.mechanism == "MNAR" and self.opt == "logistic":
            mask = self._MNAR_mask_logistic(X=X, p=self.p, p_params=self.p_obs, exclude_inputs=self.exclude_inputs)
        elif self.mechanism == "MNAR" and self.opt == "quantile":
            mask = self._MNAR_mask_quantiles(X=X, p=self.p, q=self.q, p_params=1 - self.p_obs, cut='both', MCAR=False)
        elif self.mechanism == "MNAR" and self.opt == "selfmasked":
            mask = self.MNAR_self_mask_logistic(X=X, p=self.p)
        elif self.mechanism == "MCAR":
            mask = np.random.default_rng(self.random_state).random(X.shape) < self.p
        elif self.mechanism == "PM":
            mask = np.random.default_rng(self.random_state).random((len(X), X.shape[1] -1)) < self.p
            np.insert(mask, np.random.default_rng(self.random_state).integers(low=0, high=X.shape[1] -1), False, axis=1)
        else:
            raise ValueError("MNAR mechanism can only be 'logistic', 'quantile' or 'selfmasked'")

        if self.mechanism == "PM":
            mask[:, np.random.default_rng(self.random_state).integers(low=0, high=X.shape[1])] = False

        samples_to_fix = (mask == True).all(1)
        if samples_to_fix.any():
            views_to_fix = np.random.default_rng(self.random_state).integers(low=0, high=X.shape[1], size=len(X))
            for view_idx in np.unique(views_to_fix):
                mask[samples_to_fix, view_idx] = False

            #todo assert probabilities after fixing samples
            samples_to_fix = mask.sum(1) >= 2
            mask[samples_to_fix, :]

        X[mask.astype(bool)] = np.nan
        return X

    def _MAR_mask(self, X, p, p_obs):
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)
        d_obs = max(int(p_obs * d), 1)  ## number of variables that will have no missing values (at least one variable)
        d_na = d - d_obs  ## number of variables that will have missing values
        ### Sample variables that will all be observed, and those with missing values:
        idxs_obs = np.random.default_rng(self.random_state).choice(d, d_obs, replace=False)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
        ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
        ### The parameters of this logistic model are random.
        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        coeffs = self._pick_coeffs(X, idxs_obs=idxs_obs, idxs_nas=idxs_nas)
        ### Pick the intercepts to have a desired amount of missing values
        intercepts = self._fit_intercepts(X[:, idxs_obs], coeffs, p)
        ps = np.dot(X[:, idxs_obs], coeffs) + intercepts
        ps = 1 / (1 + np.exp(-ps))
        ber = np.random.default_rng(self.random_state).random((n, d_na))
        mask[:, idxs_nas] = ber < ps
        return mask


    def _MNAR_mask_logistic(self, X, p, p_params=.3, exclude_inputs=True):
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)

        d_params = max(int(p_params * d), 1) if exclude_inputs else d  ## number of variables used as inputs (at least 1)
        d_na = d - d_params if exclude_inputs else d  ## number of variables masked with the logistic model

        ### Sample variables that will be parameters for the logistic regression:
        idxs_params = np.random.default_rng(self.random_state).choice(d,
                                                                      d_params, replace=False) if exclude_inputs else np.arange(d)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

        ### Other variables will have NA proportions selected by a logistic model
        ### The parameters of this logistic model are random.

        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        coeffs = self._pick_coeffs(X, idxs_obs=idxs_params, idxs_nas=idxs_nas)
        ### Pick the intercepts to have a desired amount of missing values
        intercepts = self._fit_intercepts(X[:, idxs_params], coeffs, p)

        ps = np.dot(X[:, idxs_params], coeffs) + intercepts
        ps = 1 / (1 + np.exp(-ps))

        ber = np.random.default_rng(self.random_state).random((n, d_na))
        mask[:, idxs_nas] = ber < ps

        ## If the inputs of the logistic model are excluded from MNAR missingness,
        ## mask some values used in the logistic model at random.
        ## This makes the missingness of other variables potentially dependent on masked values
        if exclude_inputs:
            mask[:, idxs_params] = np.random.default_rng(self.random_state).random((n, d_params)) < p

        return mask

    def MNAR_self_mask_logistic(self, X, p):
        """
        Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
        given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
        to another). The intercepts are selected to attain the desired missing rate.

        Parameters
        ----------
        X : torch.DoubleTensor or np.ndarray, shape (n, d)
            Data for which missing values will be simulated.
            If a numpy array is provided, it will be converted to a pytorch tensor.

        p : float
            Proportion of missing values to generate for variables which will have missing values.

        Returns
        -------
        mask : torch.BoolTensor or np.ndarray (depending on type of X)
            Mask of generated missing values (True if the value is missing).

        """

        n, d = X.shape
        ### Variables will have NA proportions that depend on those observed variables, through a logistic model
        ### The parameters of this logistic model are random.

        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        coeffs = self._pick_coeffs(X=X, self_mask=True)
        ### Pick the intercepts to have a desired amount of missing values
        intercepts = self._fit_intercepts(X=X, coeffs=coeffs, p=p, self_mask=True)

        ps = X*coeffs + intercepts
        ps = 1 / (1 + np.exp(-ps))

        ber = np.random.rand(n, d)
        mask = ber < ps
        return mask


    def _MNAR_mask_quantiles(self, X, p, q, p_params, cut='both', MCAR=False):
        """
        Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
        variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
        missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

        Parameters
        ----------
        X : torch.DoubleTensor or np.ndarray, shape (n, d)
            Data for which missing values will be simulated.
            If a numpy array is provided, it will be converted to a pytorch tensor.

        p : float
            Proportion of missing values to generate for variables which will have missing values.

        q : float
            Quantile level at which the cuts should occur

        p_params : float
            Proportion of variables that will have missing values

        cut : 'both', 'upper' or 'lower', default = 'both'
            Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
            in the upper quartiles of selected variables.

        MCAR : bool, default = True
            If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.

        Returns
        -------
        mask : torch.BoolTensor or np.ndarray (depending on type of X)
            Mask of generated missing values (True if the value is missing).

        """
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)

        d_na = max(int(p_params * d), 1)  ## number of variables that will have NMAR values
        ### Sample variables that will have imps at the extremes
        idxs_na = np.random.choice(d, d_na, replace=False)  ### select at least one variable with missing values

        ### check if values are greater/smaller that corresponding quantiles
        if cut == 'upper':
            quants = np.partition(a= X[:, idxs_na], kth= 1 - q, axis=0)[q]
            m = X[:, idxs_na] >= quants
        elif cut == 'lower':
            quants = np.partition(a= X[:, idxs_na], kth= q, axis=0)[q]
            m = X[:, idxs_na] <= quants
        elif cut == 'both':
            u_quants = np.partition(a= X[:, idxs_na], kth= 1 - q, axis=0)[q]
            l_quants = np.partition(a= X[:, idxs_na], kth= q, axis=0)[q]
            m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

        ### Hide some values exceeding quantiles
        ber = np.random.default_rng(self.random_state).random((n, d_na))
        mask[:, idxs_na] = (ber < p) & m

        if MCAR:
            ## Add a mcar mecanism on top
            mask = mask | (np.random.default_rng(self.random_state).random((n, d)) < p)

        return mask


    def _pick_coeffs(self, X, idxs_obs=None, idxs_nas=None, self_mask=False):
        n, d = X.shape
        if self_mask:
            coeffs = np.random.default_rng(self.random_state).normal(size=d)
            Wx = X * coeffs
            coeffs /= np.std(Wx, 0)
        else:
            d_obs = len(idxs_obs)
            d_na = len(idxs_nas)
            coeffs = np.random.default_rng(self.random_state).normal(size=(d_obs, d_na))
            Wx = np.dot(X[:, idxs_obs], coeffs)
            coeffs /= np.std(Wx, 0, keepdims=True)
        return coeffs

    def _fit_intercepts(self, X, coeffs, p, self_mask=False):
        if self_mask:
            d = len(coeffs)
            intercepts = [optimize.bisect(lambda x: (1 / (1 + np.exp(-(X * coeffs[j] + x)))).mean() - p, -50, 50) for j
                          in range(d)]
        else:
            d_obs, d_na = coeffs.shape
            intercepts = [
                optimize.bisect(lambda x: (1 / (1 + np.exp(-(np.dot(X, coeffs[:, j]) + x)))).mean() - p, -50, 50) for j
                in range(d_na)]
        return intercepts
