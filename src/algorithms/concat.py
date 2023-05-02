from .base import IMCBase
from utils import ConcatenateViews, FillMissingViews
from sklearn.preprocessing import StandardScaler


class Concat(IMCBase):
    
    
    def __init__(self, transformers = [FillMissingViews(value="mean"), ConcatenateViews(), StandardScaler().set_output(transform = 'pandas')], **args):
        self.transformers = transformers
        super().__init__(transformers = transformers, **args)
