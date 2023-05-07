from mvlearn.compose import ConcatMerger
from pipelines import BasePipeline
from transformers import ConvertToNM, FillMissingViews
from modules import NMFC


class NMFCPipeline(BasePipeline):
    
    
    def __init__(self, estimator = NMFC(n_components = 8).set_output(transform = 'pandas'),
                 transformers = [FillMissingViews(value="mean"),
                                 ConvertToNM(),
                                 ConcatMerger()], **args):

        self.estimator = estimator
        self.transformers = transformers
        super().__init__(estimator = estimator, transformers = transformers, **args)
