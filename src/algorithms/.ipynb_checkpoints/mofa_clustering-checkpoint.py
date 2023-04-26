from sklearn.pipeline import make_pipeline
from utils.utils import ConcatenateViews, FillMissingViews, ConvertToPositive
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


class NonnegativeMatrixFactorization():
    
    def __new__(cls, alg = KMeans, **args):
        
        return make_pipeline(FillMissingViews(), ConvertToPositive(), ConcatenateViews(), NMF().set_output(transform = 'pandas'), alg()).set_params(**args)