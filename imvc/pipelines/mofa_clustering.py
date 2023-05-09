from imvc.transformers import FillMissingViews
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imvc.algorithms import MOFA
from sklearn.impute import SimpleImputer
from .base import BasePipeline


class MOFAClustering(BasePipeline):
    
    def __init__(self, transformers = [FillMissingViews(value="nan"), MOFA().set_output(transform = 'pandas'), SimpleImputer(strategy='mean').set_output(transform = 'pandas'), StandardScaler().set_output(transform = 'pandas')], **args):
        
        self.transformers = transformers
        super().__init__(transformers = transformers, **args)
