from sklearn.base import BaseEstimator, ClassifierMixin
from .base import IMCBase
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import FillMissingViews, SingleView
import copy


class MSV(BaseEstimator, ClassifierMixin):
    
    def __init__(self, view_estimators = KMeans(), view_transformers = [FillMissingViews(value="mean"), SingleView(), StandardScaler().set_output(transform = 'pandas')], verbose = False):
        
        self.view_estimators = view_estimators
        self.view_transformers = view_transformers
        self.verbose = verbose
        self.pipelines_ = []

        
    def fit(self, X, y = None):
        for view_idx in range(len(X)):
            if not isinstance(self.view_estimators, list):
                if not isinstance(self.view_transformers[0], list):
                    pipeline = IMCBase(estimator = self.view_estimators, transformers = self.view_transformers, verbose = self.verbose)
                else:
                    pipeline = IMCBase(estimator = self.view_estimators, transformers = self.view_transformers[view_idx], verbose = self.verbose)
            else:
                if not isinstance(self.transformers_list[0], list):
                    pipeline = IMCBase(estimator = self.view_estimators[view_idx], transformers = self.view_transformers, verbose = self.verbose)
                else:
                    pipeline = IMCBase(estimator = self.view_estimators[view_idx], transformers = self.view_transformers[view_idx], verbose = self.verbose)
            
            self.pipelines_.append(copy.deepcopy(pipeline))
            self.pipelines_[view_idx].fit(X = X, y = y)
        return self

    def predict(self, X):
        pred = [self.pipelines_[view_idx].predict(X = X) for view_idx in range(len(X))]
        return pred