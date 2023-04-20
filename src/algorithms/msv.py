from sklearn.pipeline import make_pipeline
from utils.utils import FillMissingViews, SingleView
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import numpy as np


class MSV():
    
    def __init__(self, n_clusters : list):
        self.estimators = [make_pipeline(FillMissingViews(value = 'mean'), SingleView(view_idx = view_idx), Normalizer().set_output(transform = 'pandas'), KMeans(n_clusters = n)) for view_idx, n in enumerate(n_clusters)]
        
    def fit(self, X, y = None):
        for view_idx in range(len(X)):
            self.estimators[view_idx].fit(X = X, y = y)
        return self

    def predict(self, X):
        pred = np.array([self.estimators[view_idx].predict(X = X) for view_idx in range(len(X))])
        return pred