from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class NMFC(NMF, BaseEstimator, ClassifierMixin):
    
    
    def predict(self, X):
        transformed_X = self.transform(X)
        if not isinstance(transformed_X, pd.DataFrame):
            transformed_X = pd.DataFrame(transformed_X)
        transformed_X.columns = np.arange(transformed_X.shape[1])
        pred = transformed_X.idxmax(axis= 1)
        return pred
    
    
    def fit_predict(self, X, y = None):
        self.fit(X)
        pred = self.predict(X)
        return pred