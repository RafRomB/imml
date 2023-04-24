from sklearn.pipeline import make_pipeline
from utils.utils import ConcatenateViews, FillMissingViews
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


class Concat():
    
    def __new__(cls, alg = KMeans, **args):
        
        return make_pipeline(FillMissingViews(), ConcatenateViews(), Normalizer().set_output(transform = 'pandas'), alg()).set_params(**args)