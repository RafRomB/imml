import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import DatasetUtils


class ConvertToPositive(BaseEstimator, TransformerMixin):
    """
    Concatenate all views in a late integration before the clustering. We strongly recommend all the views have the same number of samples (you can use FillMissingViews operator in case you have an incomplete multi-view dataset).
    """
    
    def fit(self, X, y = None):
        self.negative_view_ = True if X.lt(0).any().any() else False
        return self
        
    
    def transform(self, X):
        return convert_to_positive(X = X) if self.negative_view_ else X


class FillMissingViews(BaseEstimator, TransformerMixin):
    """
    Fill missing views.
    """
    
    def __init__(self, value : str = 'mean'):
        """
        value: if 'mean', the missing features are filled based on the average of the corresponding view; if 'zeros', missing views are set as 0.
        """
        values = ['mean', 'zeros', 'nan']
        if value not in values:
            raise ValueError(f"Invalid value. Expected one of: {values}")
        self.value = value
    
    
    def fit(self, X, y=None):
        if self.value == "mean":
            self.features_view_mean_list_ = [view_data.mean() for view_idx, view_data in enumerate(X)]
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
                feautures_view_mean = self.features_view_mean_list_[view_idx]
                new_view_data = np.tile(feautures_view_mean, (n_samples,1))
            elif self.value == "zeros":
                new_view_data = np.zeros((n_samples, n_features))
            elif self.value == "nan":
                new_view_data = np.nan
            new_view_data = pd.DataFrame(new_view_data, index= missing_views.index, columns = view_data.columns)
            new_view_data[missing_views.loc[:, view_idx]] = view_data
            new_view_data = new_view_data.astype(view_data.dtypes.to_dict())
            new_X.append(new_view_data)
        return new_X

    
def convert_to_positive(X):
    positive_X = X.clip(lower = 0)
    positive_X.columns = positive_X.columns.astype(str) + '_pos'
    negative_X = 0 - X.clip(upper = 0)
    negative_X.columns = negative_X.columns.astype(str) + '_neg'
    return pd.concat([positive_X, negative_X], axis = 1)

