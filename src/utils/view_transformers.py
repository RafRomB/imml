import pandas as pd
from sklearn.preprocessing import FunctionTransformer


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
    return np.hstack(X)


def drop_view(X, view_idx : int = 0):
    return X[:view_idx] + X[view_idx+1 :]


def single_view(X, view_idx : int = 0):
    """
    ...
    """
    return X[view_idx]

