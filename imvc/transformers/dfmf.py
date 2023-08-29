import pandas as pd
from skfusion import fusion
from sklearn.base import TransformerMixin, BaseEstimator

from imvc.utils import check_Xs


class DFMF(TransformerMixin, BaseEstimator):
    r"""

    Parameters
    ----------
    factors : int, default=10
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    data_options : dict (default={})
        Data processing options, such as scale_views and scale_groups.
    data_matrix : dict (default={})
        Keys such as likelihoods, view_names, etc.
    model_options : dict (default={})
        Model options, such as ard_factors or ard_weights.
    train_options : dict (default={})
        Keys such as iter, tolerance.
    stochastic_options : dict (default={})
        Stochastic variational inference options, such as learning rate or batch size.
    covariates : dict (default={})
        Slot to store sample covariate for training in MEFISTO. Keys are sample_cov and covariates_names.
    smooth_options : dict (default={})
        options for smooth inference, such as scale_cov or model_groups.
    random_state : int (default=None)
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    mofa_ : mofa object
        Entry point as the original library. This can be used for data analysis and explainability.
    mofa_model_ : Class around HDF5-based model on disk
        This can be used for data analysis and explainability. It provides utility functions to get factors, weights,
         features, and samples info in the form of Pandas dataframes, and data as a NumPy array.
    weights_: ndarray
        Weights of the MOFA model.

    References
    ----------
    [paper1] Argelaguet R, Velten B, Arnol D, Dietrich S, Zenz T, Marioni JC, Buettner F, Huber W, Stegle O
        (2018). “Multi‐Omics Factor Analysis—a framework for unsupervised integration of multi‐omics data sets.”
        Molecular Systems Biology, 14. doi:10.15252/msb.20178124.
    [paper2] Argelaguet R, Arnol D, Bredikhin D, Deloro Y, Velten B, Marioni JC, Stegle O (2020). “MOFA+: a statistical
        framework for comprehensive integration of multi-modal single-cell data.” Genome Biology, 21.
        doi:10.1186/s13059-020-02015-1.
    [url] https://biofam.github.io/MOFA2/index.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.algorithms import MOFA
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = MOFA().fit(Xs)
    >>> transformed_Xs = pipeline.transform(Xs)
    """


    def __init__(self, factors : int = 10, args_dmfm: dict = {}, args_dmfmtransform: dict = {}):
        self.factors = factors
        self.args_dmfm = args_dmfm
        self.args_dmfmtransform = args_dmfmtransform
        self.fuser = fusion.Dfmf(**args_dmfm)
        self.transformer = fusion.DfmfTransform(**args_dmfmtransform)
        self.t = fusion.ObjectType('Type 0', factors)
        self.transform_ = None


    def fit(self, Xs, y = None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        """
        Xs = check_Xs(Xs, allow_incomplete=True, force_all_finite='allow-nan')
        self.ts = [fusion.ObjectType(f'Type {i+1}', self.factors) for i in range(len(Xs))]
        relations = [fusion.Relation(X.values, self.t, t) for X,t in zip(Xs, self.ts)]
        fusion_graph = fusion.FusionGraph(relations)
        self.fuser.fuse(fusion_graph)
        return self


    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features)
            The projected data.
        """

        Xs = check_Xs(Xs, allow_incomplete=True, force_all_finite='allow-nan')
        relations = [fusion.Relation(X.values, self.t, t) for X,t in zip(Xs, self.ts)]
        fusion_graph = fusion.FusionGraph(relations)
        transformed_X = self.transformer.transform(self.t, fusion_graph, self.fuser).factor(self.t)
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index= Xs[0].index)
        return transformed_X


    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self

