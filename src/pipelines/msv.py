from sklearn.base import BaseEstimator, ClassifierMixin
from .base import BasePipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import FillMissingViews, SingleView
import copy


class MSV(BaseEstimator, ClassifierMixin):
    
    def __init__(self, view_estimators = KMeans(), view_transformers = [FillMissingViews(value="mean"), SingleView(), StandardScaler().set_output(transform = 'pandas')], verbose = False):
        
        self.view_estimators = view_estimators
        self.view_transformers = view_transformers
        self.verbose = verbose
        self.pipelines_ = []

        
    def fit(self, X, y = None):
        for X_idx in range(len(X)):
            if not isinstance(self.view_estimators, list):
                if not isinstance(self.view_transformers[0], list):
                    pipeline = BasePipeline(estimator = self.view_estimators, transformers = self.view_transformers, verbose = self.verbose)
                else:
                    pipeline = BasePipeline(estimator = self.view_estimators, transformers = self.view_transformers[X_idx], verbose = self.verbose)
            else:
                if not isinstance(self.transformers_list[0], list):
                    pipeline = BasePipeline(estimator = self.view_estimators[X_idx], transformers = self.view_transformers, verbose = self.verbose)
                else:
                    pipeline = BasePipeline(estimator = self.view_estimators[X_idx], transformers = self.view_transformers[X_idx], verbose = self.verbose)
            
            self.pipelines_.append(copy.deepcopy(pipeline))
            self.pipelines_[X_idx].fit(X = X, y = y)
        return self

    def predict(self, X):
        pred = [self.pipelines_[X_idx].predict(X = X) for X_idx in range(len(X))]
        return pred