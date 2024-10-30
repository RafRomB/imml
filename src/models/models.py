import numpy as np
import pandas as pd
from lightning import Trainer
from lightning.pytorch.utilities.seed import isolate_rng
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from snf import compute
from torch.utils.data import DataLoader
from pyrea import clusterer, view, fuser, execute_ensemble, consensus

from imml.cluster import MRGCN
from imml.data_loader import MRGCNDataset
from imml.data_loader import DeepMFDataset

from src.utils import Utils

try:
    from rpy2.robjects.packages import importr
    rpy2_installed = True
except ImportError:
    rpy2_installed = False
    error_message = "rpy2 needs to be installed to use r engine."


class Model:
    def __init__(self, alg_name, alg):
        self.alg_name = alg_name
        self.method = alg_name.lower() if alg_name in ["SNF", "IntNMF", "COCA", "DeepMF", "Parea", "MRGCN"]\
            else "sklearn_method"
        self.method = eval(f"self.{self.method.lower()}")
        self.framework = alg_name.lower() if alg_name in ["GPCA", "AJIVE", "NMF", "DFMF", "MOFA", "jNMF"]\
            else "standard"
        self.framework = eval(f"self.{self.framework.lower()}")
        self.alg = alg


    def sklearn_method(self, train_Xs, n_clusters, random_state, run_n):
        model = self.framework(model=self.alg["alg"], n_clusters=n_clusters, random_state=random_state, run_n=run_n)
        clusters = model.fit_predict(train_Xs)
        if self.alg == "MOFA":
            transformed_Xs = model[2].factors_
            transformed_Xs = model[3].trasform(transformed_Xs)
        else:
            try:
                transformed_Xs = model[-1].embedding_
            except AttributeError:
                transformed_Xs = model[:-1].transform(train_Xs)
        return clusters, transformed_Xs


    def snf(self, train_Xs, n_clusters, random_state, run_n):
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)
        k_snf = np.ceil(len(train_Xs[0]) / 10).astype(int)
        affinities = compute.make_affinity(train_Xs, normalize=False, K=k_snf)
        fused = compute.snf(affinities, K=k_snf)
        sc = SpectralClustering(n_clusters=n_clusters, random_state=random_state + run_n)
        clusters = sc.fit_predict(fused)
        transformed_Xs = spectral_embedding(sc.affinity_matrix_, n_components=n_clusters, eigen_solver=sc.eigen_solver,
                                            random_state=sc.random_state, eigen_tol=sc.eigen_tol, drop_first=False)
        transformed_Xs = pd.DataFrame(transformed_Xs, index=train_Xs[0].index)
        return clusters, transformed_Xs


    def parea(self, train_Xs, n_clusters, random_state, run_n):
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)

        c_1_type, c_2_type, c_1_pre_type, c_2_pre_type = ('hierarchical',) *4
        c_1_method, c_1_pre_method = ('ward',) *2
        c_2_method, c_2_pre_method = ('complete',) *2
        c_1_k, c_2_k, c_1_pre_k, c_2_pre_k, k_final = (n_clusters,) *5
        fusion_method = 'disagreement'

        c1 = clusterer(c_1_type, method=c_1_method, n_clusters=c_1_k)
        c2 = clusterer(c_2_type, method=c_2_method, n_clusters=c_2_k)
        c1_pre = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=c_1_pre_k, precomputed=True)
        c2_pre = clusterer(c_2_pre_type, method=c_2_pre_method, n_clusters=c_2_pre_k, precomputed=True)
        views1 = []
        for v in train_Xs:
            views1.append(view(v, c1))
        f = fuser(fusion_method)
        v_ensemble_1 = view(execute_ensemble(views1, f), c1_pre)
        views2 = []
        for v in train_Xs:
            views2.append(view(v, c2))
        v_ensemble_2 = view(execute_ensemble(views2, f), c2_pre)
        v_res = execute_ensemble([v_ensemble_1, v_ensemble_2], f)
        if k_final:
            c_final = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=k_final, precomputed=True)
            v_res_final = view(v_res, c_final)
            labels = v_res_final.execute()

            labels = labels[:, 0]
        else:
            v1_res = view(v_res, c1_pre)
            v2_res = view(v_res, c2_pre)

            labels = consensus([v1_res.execute(), v2_res.execute()])
        transformed_Xs = pd.DataFrame(v_res, index=train_Xs[0].index)
        return labels, transformed_Xs


    def intnmf(self, train_Xs, n_clusters, random_state, run_n):
        nmf = importr("IntNMF")
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)
        clusters = nmf.nmf_mnnals(dat=Utils.convert_df_to_r_object(train_Xs),
                                  k=n_clusters, seed=int(random_state + run_n))[-1]
        clusters = np.array(clusters) - 1
        return clusters, model


    def coca(self, train_Xs, n_clusters, random_state, run_n):
        base, coca = importr("base"), importr("coca")
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)
        base.set_seed(int(random_state + run_n))
        transformed_Xs = coca.buildMOC(Utils.convert_df_to_r_object(train_Xs), M=len(train_Xs), K=n_clusters)[0]
        clusters = coca.coca(transformed_Xs, K=n_clusters)[1]
        clusters = np.array(clusters) - 1
        transformed_Xs = pd.DataFrame(np.array(transformed_Xs), index=train_Xs[0].index)
        return clusters, transformed_Xs


    def gpca(self, model, n_clusters, random_state, run_n):
        model[2].set_params(n_components=n_clusters, random_state=random_state + run_n, multiview_output=False)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def ajive(self, model, n_clusters, random_state, run_n):
        model[2].set_params(joint_rank=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def nmf(self, model, n_clusters, random_state, run_n):
        model[-3].set_params(n_components=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def deepmf(self, train_Xs, n_clusters, random_state, run_n):
        pipeline = self.alg["alg"]
        transformed_Xs = pipeline[:4].fit_transform(train_Xs)
        train_data = DeepMFDataset(X=transformed_Xs)
        train_dataloader = DataLoader(dataset=train_data, batch_size=max(128, len(transformed_Xs[0])), shuffle=True)
        trainer = Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
        transformer = pipeline[4](X=transformed_Xs)
        with isolate_rng():
            trainer.fit(transformer, train_dataloader)
        transformed_Xs = transformer.transform(transformed_Xs)
        pipeline[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        clusters = pipeline[5:].fit_predict(transformed_Xs)
        return clusters, transformed_Xs


    def mrgcn(self, train_Xs, n_clusters, random_state, run_n):
        pipeline = self.alg["alg"]
        transformed_Xs = pipeline.fit_transform(train_Xs)
        train_data = MRGCNDataset(Xs=transformed_Xs)
        with isolate_rng():
            train_dataloader = DataLoader(dataset=train_data, batch_size=max(128, len(transformed_Xs[0])), shuffle=True)
            trainer = Trainer(max_epochs=100, logger=False, enable_checkpointing=False)
            model = MRGCN(Xs=transformed_Xs, n_clusters=n_clusters)
            trainer.fit(model, train_dataloader)
        train_dataloader = DataLoader(dataset=train_data, batch_size=max(128, len(transformed_Xs[0])), shuffle=False)
        clusters = trainer.predict(model, train_dataloader)
        clusters = np.concatenate(clusters)
        transformed_Xs = [model._embedding(batch=batch).detach().cpu().numpy() for batch in train_dataloader]
        transformed_Xs = np.vstack(transformed_Xs)
        transformed_Xs = pd.DataFrame(transformed_Xs, index=train_Xs[0].index)
        return clusters, transformed_Xs


    def dfmf(self, model, n_clusters, random_state, run_n):
        model[2].set_params(n_components=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def mofa(self, model, n_clusters, random_state, run_n):
        model[2].set_params(factors=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def jnmf(self, model, n_clusters, random_state, run_n):
        model[2].set_params(n_components=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def standard(self, model, n_clusters, random_state, run_n):
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model
