from imvc.pipelines import BasePipeline
from sklearn.preprocessing import StandardScaler
from algorithms import MONET
from transformers import MultiViewTransformer


class MONETPipeline(BasePipeline):
    r"""
    Firstly fill in all the missing samples with the average features for each modality, and then concatenate all
    modal features into one. A clustering with K-means is performed then.

    Parameters
    ----------
    transformers : list of transformer, default=[FillMissingViews(value="mean"), ConcatMerger(), StandardScaler()]
        The transformers to apply to the input data before clustering. Each transformer is a transformer object
        that implements the `fit_transform` method.
    **args
        Additional parameters to pass to BasePipeline.

    References
    ----------
    [paper] Handong Zhao, Hongfu Liu, and Yun Fu. 2016. Incomplete multi-modal visual data grouping. In
        International Joint Conferences on Artificial Intelligence. 2392--2398.

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.pipelines import MONETPipeline
    >>> Xs = LoadDataset.load_incomplete_digits(p = 0.2)
    >>> pipeline = MONETPipeline()
    >>> pipeline.fit_predict(Xs)
    """

    def __init__(self, estimator= MONET(),
                 transformers=[MultiViewTransformer(transformer= StandardScaler().set_output(transform='pandas'))], **args):
        self.transformers = transformers
        self.estimator = estimator
        super().__init__(estimator=estimator, transformers=transformers, **args)
