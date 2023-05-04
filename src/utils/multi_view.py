import copy
from sklearn.base import BaseEstimator, TransformerMixin


class MultiViewTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, transformer):
        
        self.transformer = transformer
        self.transformer_list_ = []

        
    def fit(self, X, y = None):
        for view_idx in range(len(X)):
            self.transformer_list_.append(copy.deepcopy(self.transformer))
            print()
            self.transformer_list_[view_idx].fit(X = X[view_idx], y = y)
        return self

    def transform(self, X):
        return [self.transformer_list_[view_idx].transform(X = X[view_idx]) for view_idx in range(len(X))]
    
    
class MultiViewPipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, pipeline):
        
        self.pipeline = pipeline
        self.pipelines_ = []

        
    def fit(self, X, y = None):
        for view_idx, view_data in enumerate(X):
            self.pipelines_.append(copy.deepcopy(self.pipeline))
            self.pipelines_[view_idx].fit(X = view_data, y = y)
        return self

    def transform(self, X):
        return [self.pipelines_[view_idx].transform(X = view_data) for view_idx, view_data in enumerate(X)]
    
    
    def predict(self, X):
        return [self.pipelines_[view_idx].predict(X = view_data) for view_idx, view_data in enumerate(X)]
