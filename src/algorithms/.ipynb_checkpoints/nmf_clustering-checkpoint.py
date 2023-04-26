from sklearn.pipeline import make_pipeline
from utils.utils import ConcatenateViews, FillMissingViews
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


class Concat():
    
    def __new__(cls, n_clusters : int):
        return make_pipeline(FillMissingViews(value = 'mean'), ConcatenateViews(), Normalizer().set_output(transform = 'pandas'), KMeans(n_clusters = n_clusters))
