from sklearn.pipeline import make_pipeline
from utils.utils import ConcatenateViews, FillMissingViews
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class Concat():
    
    def __new__(cls, random_state : int = None, verbose = False, alg = KMeans, n_clusters : int = 8, **args):
        
        return make_pipeline(FillMissingViews(value="mean"), ConcatenateViews(), StandardScaler().set_output(transform = 'pandas'), alg(n_clusters = n_clusters, random_state = random_state, verbose = verbose)).set_params(**args)