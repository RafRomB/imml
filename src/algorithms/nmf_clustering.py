from sklearn.pipeline import make_pipeline
from utils.utils import ConcatenateViews, FillMissingViews, ConvertToPositive
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from decomposition import NMFC


class NMFClustering():
    
    def __new__(cls, n_clusters : int = 8, random_state : int = None, verbose = False, **args):
        
        return make_pipeline(FillMissingViews(value="mean"), ConvertToPositive(), ConcatenateViews(), NMFC(n_components = n_clusters, random_state = random_state, verbose = verbose).set_output(transform = 'pandas')).set_params(**args)