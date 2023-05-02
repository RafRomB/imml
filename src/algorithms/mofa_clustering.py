from utils import FillMissingViews
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from decomposition import MOFA
from sklearn.impute import SimpleImputer
from .base import IMCBase


class MOFAClustering(IMCBase):
    
    def __init__(self, transformers = [FillMissingViews(value="nan"), MOFA().set_output(transform = 'pandas'), SimpleImputer(strategy='mean').set_output(transform = 'pandas'), StandardScaler().set_output(transform = 'pandas')], **args):
        
        self.transformers = transformers
        super().__init__(transformers = transformers, **args)
