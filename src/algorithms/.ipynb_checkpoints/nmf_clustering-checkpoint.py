from .base import IMCBase
from utils import ConvertToPositive, ConcatenateViews, FillMissingViews, MultiSingleTransformer
from decomposition import NMFC


class NMFClustering(IMCBase):
    
    
    def __init__(self, estimator = NMFC(n_components = 8).set_output(transform = 'pandas'), transformers = [FillMissingViews(value="mean"), MultiSingleTransformer(transformer = ConvertToPositive()), ConcatenateViews()], **args):
        self.estimator = estimator
        self.transformers = transformers
        super().__init__(estimator = estimator, transformers = transformers, **args)
