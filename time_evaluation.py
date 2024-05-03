import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans
from mvlearn.decomposition import AJIVE, GroupPCA
from mvlearn.cluster import MultiviewSpectralClustering, MultiviewCoRegSpectralClustering
from imvc.datasets import LoadDataset
from imvc.transformers import MultiViewTransformer, ConcatenateViews
from imvc.algorithms import NMFC
from models import Model


folder_results = "results"
filelame = "time_evaluation.csv"
file_path = os.path.join(folder_results, filelame)
logs_file = os.path.join(folder_results, 'time_logs.txt')
error_file = os.path.join(folder_results, 'time_errors.txt')

random_state = 42

datasets = [
    "simulated_gm",
    "simulated_InterSIM",
    "simulated_netMUG",
    "nutrimouse_genotype",
    "nutrimouse_diet",
    "bbcsport",
    "buaa",
    "metabric",
    "digits",
    "bdgp",
    "tcga",
    "caltech101",
    "nuswide",
]
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
    "GroupPCA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), GroupPCA(), StandardScaler(), KMeans()),
                 "params": {}},
    "AJIVE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), AJIVE(),
                                   MultiViewTransformer(FunctionTransformer(pd.DataFrame)), ConcatenateViews(),
                                   StandardScaler(), KMeans()),
              "params": {}},
    "SNF": {"alg": MultiViewTransformer(StandardScaler().set_output(transform="pandas")), "params": {}},
    "IntNMF": {"alg": MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")), "params": {}},
    "COCA": {"alg": MultiViewTransformer(StandardScaler().set_output(transform="pandas")), "params": {}},
}

if os.path.exists(file_path):
    results = pd.read_csv(file_path, index_col=0)
else:
    results = pd.DataFrame(0, index=datasets, columns=algorithms.keys())
    os.remove(logs_file) if os.path.exists(logs_file) else None
    os.remove(error_file) if os.path.exists(error_file) else None
    open(logs_file, 'w').close()
    open(error_file, 'w').close()

errors_dict = defaultdict(int)

for dataset_name in datasets:
    names = dataset_name.split("_")
    if "simulated" in names:
        names = ["_".join(names)]
    x_name, y_name = names if len(names) > 1 else (names[0], "0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle=True)
    y = y[y_name]
    n_clusters = y.nunique()

    for alg_name, alg in algorithms.items():
        time_execution = results.loc[dataset_name, alg_name]
        if (time_execution > 0) or np.isnan(time_execution):
            continue
        with open(logs_file, "a") as f:
            f.write(f'\n {dataset_name} \t {alg_name} \t {datetime.now()}')

        try:
            start_time = time.perf_counter()
            clusters, _ = Model(alg_name=alg_name, alg=alg).method(train_Xs=Xs, n_clusters=n_clusters,
                                                                       random_state=random_state, run_n=0)
            elapsed_time = time.perf_counter() - start_time
        except Exception as exception:
            errors_dict[f"{type(exception).__name__}: {exception}"] += 1
            with open(error_file, "a") as f:
                f.write(
                    f'\n {dataset_name} \t {alg_name} \t {errors_dict} \t {datetime.now()}')
            elapsed_time = np.nan

        results.loc[dataset_name, alg_name] = elapsed_time
        results.to_csv(os.path.join(folder_results, filelame))


