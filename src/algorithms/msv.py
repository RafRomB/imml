from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from utils.utils import FillMissingViews, SingleView
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


class MSV(BaseEstimator, ClassifierMixin):
    
    def __init__(self, random_state : int = None, verbose = False, alg = KMeans, n_clusters : int = 8, **args):
        
        self.estimators = [make_pipeline(FillMissingViews(value="mean"), SingleView(view_idx = view_idx), StandardScaler().set_output(transform = 'pandas'), alg(n_clusters = n_clusters, random_state = random_state, verbose = verbose)).set_params(**args) for view_idx, n in enumerate(n_clusters)]
        
    def fit(self, X, y = None):
        for view_idx in range(len(X)):
            self.estimators[view_idx].fit(X = X, y = y)
        return self

    def predict(self, X):
        pred = np.array([self.estimators[view_idx].predict(X = X) for view_idx in range(len(X))])
        return pred