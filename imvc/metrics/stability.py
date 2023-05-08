from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class StabilityMetrics():
    
    
    def __init__(self, X, estimator = None, n_times : int = 100, feature_frac : float = 0.7, random_state : int = None):
        self.boost_estimator = make_pipeline(FunctionTransformer(lambda x: [view[view.columns.to_series().sample(frac = feature_frac, random_state = random_state)] for view in x]), *estimator)
        self.feature_frac = feature_frac
        self.X = X
        self.transformed_X = estimator[:-1].fit_transform(X)
        self.n_times = n_times
        self.labels_ = np.sort(np.unique(estimator[-1].labels_))
        self.distances = pd.DataFrame(euclidean_distances(self.transformed_X))
        self.boost_preds = pd.DataFrame([self.boost_estimator.fit_predict(self.X) for time in range(self.n_times)]).transpose()
        self.nc1 = len(self.labels_)
        
    def compute(self, pred):
        metrics = {'APN': [], 'AD': [], 'ADM': [], 'FOM': []}
        dij = np.full((self.nc1, self.nc1), np.nan)
        dij2 = np.zeros((self.nc1, self.nc1))
        xbar = self.transformed_X.groupby(pred).mean().mean(1)
        for iter_boost in self.boost_preds:
            boost_pred = self.boost_preds[iter_boost]
            overlap = pd.crosstab(index = pred, columns = boost_pred)
            xbar_boost = self.transformed_X.groupby(pred).mean().mean(1)
            for idx, cluster in enumerate(self.labels_):
                xbari = xbar[idx]
                for idx_boost, cluster_boost in enumerate(self.labels_):
                    clus = self.transformed_X.index[pred == cluster]
                    clus_boost = self.transformed_X.index[boost_pred == cluster_boost]
                    cl = len(clus)*len(clus_boost)
                    if (len(clus)*len(clus_boost)) >0:
                        dij[idx,idx_boost] = np.mean(self.distances.iloc[clus, clus_boost])
                    diff = xbari - xbar_boost
                    if len(diff)>0:
                        dij2[idx,idx_boost] = np.sqrt(np.sum(diff**2))
            rs = overlap.sum(axis=1)
            metrics['APN'].append(1 - sum(overlap**2/rs)/sum(overlap))
            metrics['AD'].append(sum(overlap*dij)/len(self.transformed_X))
            metrics['ADM'].append(sum(overlap*dij2)/len(self.transformed_X))
            xbar = self.transformed_X.groupby(boost_pred).mean().mean(1)
            subs = boost_pred.to_frame().apply(lambda x: xbar.loc[x], axis = 0).squeeze().reset_index(drop=True)
            metrics["FOM"].append(np.sqrt(np.mean((self.transformed_X.subtract(subs, axis = 0))**2)) / np.sqrt((self.transformed_X.shape[0] - self.nc1) / self.transformed_X.shape[0]))
        metrics = {key: [np.mean(value), np.std(value), value] for key,value in metrics.items()}
        return metrics

        
    def apn(self, pred):
        metric = []
        dij = np.full((self.nc1, self.nc1), np.nan)
        dij2 = np.zeros((self.nc1, self.nc1))
        xbar = self.transformed_X.groupby(pred).mean().mean(1)
        for iter_boost in self.boost_preds:
            boost_pred = self.boost_preds[iter_boost]
            overlap = pd.crosstab(index = pred, columns = boost_pred)
            rs = overlap.sum(axis=1)
            metric.append(1 - sum(overlap**2/rs)/sum(overlap))
        metric = [np.mean(metric), np.std(metric), metric]
        return metric

        
    def ad(self, pred):
        metric = []
        dij = np.full((self.nc1, self.nc1), np.nan)
        xbar = self.transformed_X.groupby(pred).mean().mean(1)
        for iter_boost in self.boost_preds:
            boost_pred = self.boost_preds[iter_boost]
            overlap = pd.crosstab(index = pred, columns = boost_pred)
            xbar_boost = self.transformed_X.groupby(pred).mean().mean(1)
            for idx, cluster in enumerate(self.labels_):
                xbari = xbar[idx]
                for idx_boost, cluster_boost in enumerate(self.labels_):
                    clus = self.transformed_X.index[pred == cluster]
                    clus_boost = self.transformed_X.index[boost_pred == cluster_boost]
                    cl = len(clus)*len(clus_boost)
                    if (len(clus)*len(clus_boost)) >0:
                        dij[idx,idx_boost] = np.mean(self.distances.iloc[clus, clus_boost])
            rs = overlap.sum(axis=1)
            metric.append(sum(overlap*dij)/len(self.transformed_X))
        metric = [np.mean(metric), np.std(metric), metric]
        return metric

        
    def adm(self, pred):
        metric = []
        dij2 = np.zeros((self.nc1, self.nc1))
        xbar = self.transformed_X.groupby(pred).mean().mean(1)
        for iter_boost in self.boost_preds:
            boost_pred = self.boost_preds[iter_boost]
            overlap = pd.crosstab(index = pred, columns = boost_pred)
            xbar_boost = self.transformed_X.groupby(pred).mean().mean(1)
            for idx, cluster in enumerate(self.labels_):
                xbari = xbar[idx]
                for idx_boost, cluster_boost in enumerate(self.labels_):
                    diff = xbari - xbar_boost
                    if len(diff)>0:
                        dij2[idx,idx_boost] = np.sqrt(np.sum(diff**2))
            rs = overlap.sum(axis=1)
            metric.append(sum(overlap*dij2)/len(self.transformed_X))
        metric = [np.mean(metric), np.std(metric), metric]
        return metric

        
    def fom(self, pred):
        metric = []
        xbar = self.transformed_X.groupby(pred).mean().mean(1)
        for iter_boost in self.boost_preds:
            boost_pred = self.boost_preds[iter_boost]
            xbar = self.transformed_X.groupby(boost_pred).mean().mean(1)
            subs = boost_pred.to_frame().apply(lambda x: xbar.loc[x], axis = 0).squeeze().reset_index(drop=True)
            metric.append(np.sqrt(np.mean((self.transformed_X.subtract(subs, axis = 0))**2)) / np.sqrt((self.transformed_X.shape[0] - self.nc1) / self.transformed_X.shape[0]))
        metric = [np.mean(metric), np.std(metric), metric]
        return metric

