from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn import metrics


class ExternalValidation():
    
    
    def __init__(self, X, estimator, sample_views, n_times : int = 100, sample_frac : float = 0.7, random_state : int = None):
        self.estimator = estimator
        self.sample_views = sample_views
        self.sample_frac = sample_frac
        self.X = X
        self.n_times = n_times
        self.random_state = random_state
        
        train_labels, val_labels, train_pred = [], [], []
        val_labels = []
        
        for train_index, val_index in ShuffleSplit(n_splits=self.n_times, test_size = sample_frac, random_state = random_state).split(sample_views):
            train_index = sample_views.iloc[train_index].index
            val_index = sample_views.iloc[val_index].index
            train_estimator = make_pipeline(FunctionTransformer(lambda x: [view.loc[view.index.intersection(train_index)] for view in x]), *estimator)
            val_estimator = make_pipeline(FunctionTransformer(lambda x: [view.loc[view.index.intersection(val_index)] for view in x]), *estimator)
            train_labels.append(pd.Series(train_estimator.fit_predict(X), index = sample_views.iloc[train_index].index))
            val_labels.append(pd.Series(val_estimator.fit_predict(X), index = sample_views.iloc[val_index].index))
            train_pred.append(pd.Series(train_estimator.predict(X), index = sample_views.iloc[val_index].index))
            
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.train_pred = train_pred
        
        
    def compute_stability(self):
        metric_df = pd.Series(np.arange(self.n_times))
        metric_df = metric_df.apply(lambda x: {
            'AMI': metrics.adjusted_mutual_info_score(labels_true = self.val_labels[x], labels_pred = self.train_pred[x]),
            'ARI': metrics.adjusted_rand_score(labels_true = self.val_labels[x], labels_pred = self.train_pred[x]),
            'Completeness': metrics.completeness_score(labels_true = self.val_labels[x], labels_pred = self.train_pred[x]),
            'FMI': metrics.fowlkes_mallows_score(labels_true = self.val_labels[x], labels_pred = self.train_pred[x])
        })
        metric_df = pd.DataFrame(list(metric_df))
        return metric_df.mean(), metric_df.std(), metric_df
        


        
