from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans


class AlgBase(BaseEstimator, ClassifierMixin):
    
    def __init__(self, clustering_estimator = None, n_clusters : int = None):
        super().__init__()
        
        if clustering_estimator is None:
            clustering_estimator = KMeans
        if n_clusters is not None:
            args = {"n_clusters" : n_clusters}
            clustering_estimator = KMeans(**args)
        else:
            clustering_estimator = KMeans(random_state=42)
        self.estimator = clustering_estimator
        
        
