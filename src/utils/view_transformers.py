import pandas as pd
from sklearn.preprocessing import FunctionTransformer

    
class ConcatenateViews(FunctionTransformer):
    """
    Concatenate all views in a late integration before the clustering. We strongly recommend all the views have the same number of samples (you can use FillMissingViews operator in case you have an incomplete multi-view dataset).
    """
    
    def __init__(self):
        super().__init__(concatenate_views)


class DropView(FunctionTransformer):
    """
    ...
    """
    
    def __init__(self, view_idx : int = 0):
        self.view_idx = view_idx
        super().__init__(drop_view, kw_args = {"view_idx": view_idx})

        
class SingleView(FunctionTransformer):
    """
    ...
    """
    
    def __init__(self, view_idx : int = 0):
        self.view_idx = view_idx
        super().__init__(single_view, kw_args = {"view_idx": view_idx})
        
        
def concatenate_views(X):
    """
    ...
    """
    return pd.concat(X, axis = 1)


def drop_view(X, view_idx : int = 0):
    return X[:view_idx] + X[view_idx+1 :]


def single_view(X, view_idx : int = 0):
    """
    ...
    """
    return X[view_idx]

