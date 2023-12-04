import os.path
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, spectral_clustering
from mvlearn.decomposition import AJIVE, GroupPCA
from mvlearn.cluster import MultiviewSpectralClustering, MultiviewCoRegSpectralClustering
from snf import compute
from bignmf.models.jnmf.integrative import IntegrativeJnmf
from bignmf.models.jnmf.standard import StandardJnmf
from imvc.datasets import LoadDataset
from imvc.utils import DatasetUtils
from imvc.transformers import MultiViewTransformer, ConcatenateViews
from imvc.algorithms import NMFC

from utils.utils import save_record

folder_name = "results"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_name, filelame)

random_state = 42
START_BENCHMARKING = False

datasets = ["nutrimouse_genotype", "nutrimouse_diet", "bbcsport", "bdgp", "caltech101", "digits", "tcga_tissue", "tcga_survival", "nuswide", "metabric"]
probs = np.arange(100, step= 10)
imputation = [True, False]
runs_per_alg = np.arange(10)
algorithms = {
    "Concat": {"alg": make_pipeline(ConcatenateViews(),
                                    StandardScaler().set_output(transform='pandas'),
                                    KMeans()), "params": {}},
    "NMFC": {"alg": make_pipeline(ConcatenateViews(),
                                  MinMaxScaler().set_output(transform='pandas'),
                                  NMFC().set_output(transform='pandas')), "params": {}},
    "MVSpectralClustering": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                                  MultiviewSpectralClustering()),
                             "params": {}},
    "MVCoRegSpectralClustering": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                                       MultiviewCoRegSpectralClustering()),
                                  "params": {}},
    "GroupPCA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                      GroupPCA(), StandardScaler().set_output(transform= "pandas"), KMeans()),
                 "params": {}},
    "AJIVE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                   AJIVE(), ConcatenateViews(), StandardScaler().set_output(transform= "pandas"),
                                   KMeans()),
              "params": {}},
    "SNF": {},
    "intNMF": {},
    "jNMF": {},
}
indexes_results = {"dataset": datasets, "algorithm": list(algorithms.keys()),
                   "missing_percentage": probs, "imputation": imputation, "run_n": runs_per_alg}


if START_BENCHMARKING:
    results = pd.DataFrame(datasets, columns= ["dataset"])
    for k,v in {k:v for k,v in indexes_results.items() if k != "dataset"}.items():
        results = results.merge(pd.Series(v, name= k), how= "cross")
    results = results.set_index(list(indexes_results.keys()))
    results["finished"] = False
else:
    results = pd.read_csv(file_path, index_col= list(indexes_results.keys()))

unfinished_results = results.loc[~results["finished"]]

for dataset_name in unfinished_results.index.get_level_values("dataset").unique():
    Xs, y = LoadDataset.load_dataset(dataset_name=dataset_name.split("_")[0], return_y=True, shuffle= False)
    y = pd.DataFrame(y)
    for target in y.columns:
        y_series = y[target].squeeze()
        n_clusters = y_series.nunique()

        for idx_iterator in unfinished_results.loc[unfinished_results.index.get_level_values("dataset") == dataset_name].itertuples():
            idx = idx_iterator[0]
            row = results.loc[[idx]]
            row_index = row.index
            print(row.drop(columns= row.columns).reset_index().to_dict(orient="records")[0])
            alg_name, impute, p, run_n = (
                row_index.get_level_values("algorithm")[0],
                row_index.get_level_values("imputation")[0],
                row_index.get_level_values("missing_percentage")[0]/100,
                row_index.get_level_values("run_n")[0])

            alg = algorithms[alg_name]
            train_Xs = DatasetUtils.shuffle_imvd(Xs=Xs, random_state= random_state + run_n)
            y_train = y_series.loc[train_Xs[0].index]
            if p != 0:
                if n_clusters > len(train_Xs[0])*p:
                    continue
                try:
                    train_Xs = DatasetUtils.add_random_noise_to_views(Xs=train_Xs, p= round(p, 2),
                                                                      random_state =random_state + run_n, 
                                                                      assess_percentage = True, stratify = y_train)
                except Exception as ex:
                    print(ex)
                    continue

            if impute:
                train_Xs = MultiViewTransformer(SimpleImputer(strategy="mean").set_output(transform= "pandas")).fit_transform(train_Xs)
            else:
                train_Xs = DatasetUtils.select_complete_samples(Xs = train_Xs)
                y_train = y_train.loc[train_Xs[0].index]

            errors_dict = defaultdict(int)
            start_time = time.perf_counter()
            if alg_name == "SNF":
                preprocessing_step = MultiViewTransformer(StandardScaler().set_output(transform= "pandas"))
                train_Xs = preprocessing_step.fit_transform(train_Xs)
                affinities = compute.make_affinity(train_Xs, normalize= False)
                fused = compute.snf(affinities)
                clusters = spectral_clustering(fused, n_clusters=n_clusters, random_state=random_state + run_n)
            elif alg_name == "intNMF":
                preprocessing_step = MultiViewTransformer(MinMaxScaler().set_output(transform= "pandas"))
                model = IntegrativeJnmf({k:v for k,v in enumerate(train_Xs)}, k= n_clusters, lamb = 0.1)
                model.run(trials = 50, iterations = 100, verbose=0)
                model.cluster_data()
                clusters = model.w_cluster
            elif alg_name == "jNMF":
                pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform= "pandas")))
                model = StandardJnmf({k:v for k,v in enumerate(train_Xs)}, k= n_clusters)
                model.run(trials = 50, iterations = 100, verbose=0)
                model.cluster_data()
                clusters = model.w_cluster
            else:
                model, params = alg["alg"], alg["params"]
                if alg_name == "GroupPCA":
                    model[-3].set_params(n_components=n_clusters, random_state=random_state + run_n, multiview_output=False)
                elif alg_name == "AJIVE":
                    model[-4].set_params(random_state=random_state + run_n)
                if alg_name == "NMFC":
                    model[-1].set_params(n_components=n_clusters, random_state=random_state + run_n)
                else:
                    model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
                clusters = model.fit_predict(train_Xs)
            clusters = pd.Series(clusters, index= y_train.index)

            elapsed_time = time.perf_counter() - start_time

            if alg_name in ["NMFC"]:
                train_X = model.transform(train_Xs)
            elif alg_name in ["SNF"]:
                train_X = preprocessing_step.transform(train_Xs)
            elif alg_name in ["intNMF", "jNMF"]:
                train_X = model.w
            else:
                train_X = model[:-1].transform(train_Xs)
            if isinstance(train_X, list):
                train_X = ConcatenateViews().fit_transform(train_X)

            if p > 0:
                best_solution = pd.MultiIndex.from_arrays([[row_index.get_level_values(level= level)[0]] if level != "missing_percentage" else [0]
                                                           for level in row_index.names], names= row_index.names)
                best_solution = results.loc[best_solution]["y_pred"][0]
            else:
                best_solution = None

            dict_results = save_record(train_Xs=train_Xs, train_X=train_X, clusters=clusters, y=y_train, p= p,
                              best_solution = best_solution, elapsed_time=elapsed_time,
                              random_state=random_state, errors_dict=errors_dict)
            dict_results = pd.DataFrame(pd.Series(dict_results), columns= row_index).T
            results.loc[[idx], dict_results.columns] = dict_results
            results.loc[idx, "finished"] = True
            results.to_csv(file_path)

