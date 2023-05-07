from mvlearn.compose import ConcatMerger
from .base import BasePipeline
from transformers import FillMissingViews
from sklearn.preprocessing import StandardScaler


class Concat(BasePipeline):
    
    
    def __init__(self, transformers = [FillMissingViews(value="mean"), ConcatMerger(), StandardScaler().set_output(transform = 'pandas')], **args):
        self.transformers = transformers
        super().__init__(transformers = transformers, **args)
