from sklearn.pipeline import make_pipeline
from utils.utils import FillMissingViews
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from decomposition import MOFA
from sklearn.impute import SimpleImputer


class MOFAClustering():
    
    def __new__(cls, factors : int = 10, random_state : int = None, verbose = False, alg = KMeans, n_clusters : int = 8, **args):
        
        return make_pipeline(FillMissingViews(value="nan"), MOFA(factors = factors, random_state = random_state, verbose = verbose).set_output(transform = 'pandas'), SimpleImputer(strategy='mean'), StandardScaler(), alg(n_clusters = n_clusters, random_state = random_state, verbose = verbose)).set_params(**args)