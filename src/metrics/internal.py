import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score


class InternalMetrics():
    
    
    def __init__(self, X, estimator = None, n_neighbors : int = 10, random_state : int = None):
        if estimator is not None:
            self.X = estimator[:-1].fit_transform(X)
        else:
            self.X = X
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    
    def __delta(self, ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]
        return np.min(values)

    
    def __big_delta(self, ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        return np.max(values)

    
    def dunn_index(self, labels_pred, distances = None):
        """
        Adapted from: https://github.com/jqmviegas/jqm_cvi
        """
        if distances is None:
            distances = euclidean_distances(self.X)
        ks = np.sort(np.unique(labels_pred))

        deltas = np.ones([len(ks), len(ks)])*1000000
        big_deltas = np.zeros([len(ks), 1])
        l_range = list(range(0, len(ks)))

        for k in l_range:
            for l in (l_range[0:k]+l_range[k+1:]):
                deltas[k, l] = self.__delta((labels_pred == ks[k]), (labels_pred == ks[l]), distances)

            big_deltas[k] = self.__big_delta((labels_pred == ks[k]), distances)

        di = np.min(deltas)/np.max(big_deltas)
        return di
    
    
    def connectivity(self, labels_pred, distances = None, n_neighbors=10):
        if distances is None:
            distances = euclidean_distances(self.X)
        nearest = np.apply_along_axis(lambda x: np.argsort(x)[1:(n_neighbors+1)], 1, distances)
        nr, nc = nearest.shape
        clusters_mat = np.tile(labels_pred, (nc, 1)).T
        same = (clusters_mat != labels_pred[nearest])
        conn = np.sum(same * np.tile(1/np.arange(1, n_neighbors+1), (nr, 1)))
        return conn
    
    
    def compute(self, labels_pred):
        metrics = {}
        distances = euclidean_distances(self.X)
        dunn_m = self.dunn_index(labels_pred = labels_pred, distances = distances)
        metrics['Dunn'] = dunn_m
        sil_m = silhouette_score(self.X, labels = labels_pred, random_state = self.random_state)
        metrics['Silhouette'] = sil_m
        conn_m = self.connectivity(labels_pred = labels_pred, distances = distances, n_neighbors = self.n_neighbors)
        metrics['Connectivity'] = conn_m
        return metrics
        

