import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.preprocessing import StandardScaler
from snf import compute

from imvc.transformers import MultiViewTransformer


class Model:
    def __init__(self, alg_name, alg):
        self.alg_name = alg_name.lower() if alg_name in ["GroupPCA", "AJIVE", "NMFC"] else "standard"
        self.method = alg_name.lower() if alg_name in ["SNF"] else "sklearn_method"
        self.alg_name = eval(f"self.{self.alg_name.lower()}")
        self.method = eval(f"self.{self.method.lower()}")
        self.alg = alg


    def sklearn_method(self, train_Xs, y_train, n_clusters, random_state, run_n):
        model, params = self.alg["alg"], self.alg["params"]
        model = self.alg_name(model=model, n_clusters=n_clusters, random_state=random_state, run_n=run_n)
        clusters = model.fit_predict(train_Xs)
        return clusters, model


    def snf(self, train_Xs, y_train, n_clusters, random_state, run_n):
        preprocessing_step = MultiViewTransformer(StandardScaler().set_output(transform="pandas"))
        train_Xs = preprocessing_step.fit_transform(train_Xs)
        k_snf = np.ceil(len(y_train) / 10).astype(int)
        affinities = compute.make_affinity(train_Xs, normalize=False, K=k_snf)
        fused = compute.snf(affinities, K=k_snf)
        clusters = spectral_clustering(fused, n_clusters=n_clusters, random_state=random_state + run_n)
        return clusters, preprocessing_step


    def grouppca(self, model, n_clusters, random_state, run_n):
        model[1].set_params(n_components=n_clusters, random_state=random_state + run_n, multiview_output=False)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def ajive(self, model, n_clusters, random_state, run_n):
        model[1].set_params(random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def nmfc(self, model, n_clusters, random_state, run_n):
        model[-1].set_params(n_components=n_clusters, random_state=random_state + run_n)
        return model


    def standard(self, model, n_clusters, random_state, run_n):
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model
