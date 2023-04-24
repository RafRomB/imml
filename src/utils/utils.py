import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_blobs


class ConvertToPositive(BaseEstimator, TransformerMixin):
    """
    Concatenate all views in a late integration before the clustering. We strongly recommend all the views have the same number of samples (you can use FillMissingViews operator in case you have an incomplete multi-view dataset).
    """
    
    def __new__(cls):
        return FunctionTransformer(lambda x: [convert_to_positive(X = view) if view.lt(0).any().any() else view for view in x])

    
class ConcatenateViews(BaseEstimator, TransformerMixin):
    """
    Concatenate all views in a late integration before the clustering. We strongly recommend all the views have the same number of samples (you can use FillMissingViews operator in case you have an incomplete multi-view dataset).
    """
    
    def __new__(cls):
        return FunctionTransformer(lambda x: pd.concat(x, axis = 1))


class DropView(BaseEstimator, TransformerMixin):
    """
    ...
    """
    
    def __new__(cls, view_idx : int):
        return FunctionTransformer(lambda x: x[:view_idx] + x[view_idx+1 :])


class SingleView(BaseEstimator, TransformerMixin):
    """
    ...
    """
    
    def __new__(cls, view_idx : int):
        return FunctionTransformer(lambda x: x[view_idx])


class FillMissingViews(BaseEstimator, TransformerMixin):
    """
    Fill missing views.
    """
    
    def __init__(self, value : str = 'mean'):
        """
        value: if 'mean', the missing features are filled based on the average of the corresponding view; if 'zeros', missing views are set as 0.
        """
        values = ['mean', 'zeros']
        if value not in values:
            raise ValueError(f"Invalid value. Expected one of: {values}")
        self.value = value
    
    
    def fit(self, X, y=None):
        if self.value == "mean":
            self.features_view_mean_list = [view_data.mean() for view_idx, view_data in enumerate(X)]
        elif self.value == "zeros":
            pass
        return self

    
    def transform(self, X, y=None):
        sample_views = get_sample_views(imvd = X)
        missing_views = sample_views == 1
        n_samples = len(missing_views)
        
        new_X = []
        for view_idx, view_data in enumerate(X):
            n_features = view_data.shape[1]
            if self.value == "mean":
                feautures_view_mean = self.features_view_mean_list[view_idx]
                new_view_data = np.tile(feautures_view_mean, (n_samples,1))
            elif self.value == "zeros":
                new_view_data = np.zeros((n_samples, n_features))
            new_view_data = pd.DataFrame(new_view_data, index= missing_views.index, columns = view_data.columns)
            new_view_data = new_view_data.astype(view_data.dtypes.to_dict())
            new_view_data[missing_views.loc[:, view_idx]] = view_data
            new_X.append(new_view_data)
        return new_X
    
    
class RandomMultiviewDataset:
    def __init__(self):
        pass
    
    def create_mvd(self, n_samples : int = 100, n_features : list = [10, 50, 100], n_clusters : list = [3, 4, 2]):
        mvd = [pd.DataFrame(make_blobs(n_samples=n_samples, n_features=n, centers=centers)[0]) for centers, n in zip(n_clusters, n_features)]
        return mvd
    
    
    def add_sample_views(self, mvd : list, p : list = [0.2, 0.3, 0.5]):
        n_samples = len(mvd[0])
        sample_views = []
        for idx,view in enumerate(mvd):
            sample_view = np.random.choice([0, 1], size= n_samples, p=[p[idx], 1 - p[idx]]).tolist()
            sample_views.append(sample_view)
        sample_views = pd.DataFrame(sample_views, columns = view.index).transpose()
        sample_views[sample_views.sum(1) == 0] = 1
        return sample_views
    
    
    def add_missing_views(self, mvd : list, sample_views : pd.DataFrame):
        missing_views = sample_views == 1
        for view_idx, view_data in enumerate(mvd):
            mvd[view_idx] = view_data[missing_views.loc[:, view_idx]]
        return mvd

    
def get_sample_views(imvd : list):
    sample_views = pd.concat([view.index.to_series() for view in imvd], axis = 1).sort_index()
    sample_views = sample_views.mask(sample_views.isna(), 0).where(sample_views.isna(), 1).astype(int)
    return sample_views

    
def convert_to_positive(X, y=None):

    positive_X = X.clip(lower = 0)
    positive_X.columns = positive_X.columns.astype(str) + '_pos'
    negative_X = 0 - X.clip(upper = 0)
    negative_X.columns = negative_X.columns.astype(str) + '_neg'
    return pd.concat([positive_X, negative_X], axis = 1)
