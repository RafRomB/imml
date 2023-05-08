import pandas as pd
from mvlearn.datasets import load_nutrimouse

from imvc.utils import DatasetUtils


def load_incomplete_nutrimouse(p: list):
    r"""
    Load an incomplete multi-view version of the Nutrimouse dataset [#1paper]_, a two-view dataset from a nutrition
    study on mice, as available from https://CRAN.R-project.org/package=CCA [#2r]_.

    Parameters
    ----------
    p: list or int
        The percentaje that each view will have for missing samples. If p is int, all the views will have the
        same percentaje.

    Returns
    -------
    Xs : list of array-likes
        - Xs length: n_views
        - Xs[i] shape: (n_samples_i, n_features_i)
        A list of different views.

    Notes
    -----
    This data consists of two views from a nutrition study of 40 mice:
    - gene : expressions of 120 potentially relevant genes
    - lipid : concentrations of 21 hepatic fatty acids

    References
    ----------
    .. [#1paper] P. Martin, H. Guillou, F. Lasserre, S. Déjean, A. Lan, J-M.
            Pascussi, M. San Cristobal, P. Legrand, P. Besse, T. Pineau.
            "Novel aspects of PPARalpha-mediated regulation of lipid and
            xenobiotic metabolism revealed through a nutrigenomic study."
            Hepatology, 2007.
    .. [#2r] González I., Déjean S., Martin P.G.P and Baccini, A. (2008) CCA:
            "An R Package to Extend Canonical Correlation Analysis." Journal
            of Statistical Software, 23(12).

     Examples
    --------
    >>> from imvc.datasets import load_incomplete_nutrimouse
    >>> load_incomplete_nutrimouse(p = [0.2, 0.5])
   """
    dict_data = load_nutrimouse()
    gene = pd.DataFrame(dict_data['gene'], columns=dict_data['gene_feature_names'])
    lipid = pd.DataFrame(dict_data['lipid'], columns=dict_data['lipid_feature_names'])
    Xs = [gene, lipid]
    Xs = DatasetUtils().create_imvd_from_mvd(Xs=Xs, p=p)
    return Xs
