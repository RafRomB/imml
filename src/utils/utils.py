import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_blobs


class DatasetUtils():
    
    @staticmethod
    def add_sample_views(mvd : list, p : list = [0.2, 0.3, 0.5]):
        n_samples = len(mvd[0])
        sample_views = []
        for idx,view in enumerate(mvd):
            sample_view = np.random.choice([0, 1], size= n_samples, p=[p[idx], 1 - p[idx]]).tolist()
            sample_views.append(sample_view)
        sample_views = pd.DataFrame(sample_views, columns = view.index).transpose()
        sample_views[sample_views.sum(1) == 0] = 1
        return sample_views
    
    @staticmethod
    def add_missing_views(mvd : list, sample_views : pd.DataFrame):
        imvd = mvd.copy()
        missing_views = sample_views == 1
        for view_idx, view_data in enumerate(mvd):
            imvd[view_idx] = view_data[missing_views.loc[:, view_idx]]
        return imvd

    @staticmethod
    def get_sample_views(imvd : list):
        sample_views = pd.concat([view.index.to_series() for view in imvd], axis = 1).sort_index()
        sample_views = sample_views.mask(sample_views.isna(), 0).where(sample_views.isna(), 1).astype(int)
        return sample_views





