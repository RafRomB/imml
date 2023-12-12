import os.path
import argparse
import numpy as np
from pandarallel import pandarallel
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans
from mvlearn.decomposition import AJIVE, GroupPCA
from mvlearn.cluster import MultiviewSpectralClustering, MultiviewCoRegSpectralClustering
from imvc.datasets import LoadDataset
from imvc.transformers import MultiViewTransformer, ConcatenateViews
from imvc.algorithms import NMFC

from utils import Utils


folder_results = "results"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_results, filelame)
logs_file = os.path.join(folder_results, 'logs.txt')
error_file = os.path.join(folder_results, 'error.txt')

random_state = 42

parser = argparse.ArgumentParser()
parser.add_argument('-continue_benchmarking', default= False, action='store_true')
parser.add_argument('-n_jobs', default= 1, type= int)
args = parser.parse_args()

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

datasets = [
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
    "GroupPCA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), GroupPCA(), StandardScaler(), KMeans()),
                 "params": {}},
    "AJIVE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), AJIVE(), MultiViewTransformer(FunctionTransformer(pd.DataFrame)),
                                   ConcatenateViews(), StandardScaler(), KMeans()),
              "params": {}},
    "SNF": {},
    "intNMF": {},
    "jNMF": {},
}
indexes_results = {"dataset": datasets, "algorithm": list(algorithms.keys()),
                   "missing_percentage": probs, "imputation": imputation, "run_n": runs_per_alg}

if not args.continue_benchmarking:
    results = pd.DataFrame(datasets, columns= ["dataset"])
    for k,v in {k:v for k,v in indexes_results.items() if k != "dataset"}.items():
        results = results.merge(pd.Series(v, name= k), how= "cross")
    results = results.set_index(list(indexes_results.keys()))
    idx_to_drop = results.xs(0, level="missing_percentage",
                             drop_level=False).xs(True, level="imputation", drop_level=False).index
    results = results.drop(idx_to_drop)
    results[["finished", "completed"]] = False

    os.remove(logs_file) if os.path.exists(logs_file) else None
    os.remove(error_file) if os.path.exists(error_file) else None
    open(logs_file, 'w').close()
    open(error_file, 'w').close()
else:
    results = pd.read_csv(file_path, index_col= list(indexes_results.keys()))
    drop_columns = ["comments", "stratified"] if "stratified" in results.select_dtypes(object).columns else "comments"
    results_ = results.select_dtypes(object).drop(columns= drop_columns).replace(np.nan, "np.nan")
    for col in results_.columns:
        results[col] = results_[col].apply(eval)

unfinished_results = results.loc[~results["finished"]]

for dataset_name in unfinished_results.index.get_level_values("dataset").unique():
    names = dataset_name.split("_")
    if "simulated" in names:
        names = ["_".join(names)]
    x_name,y_name = names if len(names) > 1 else (names[0], "0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle= False)
    y = y[y_name]
    n_clusters = y.nunique()

    if args.n_jobs == 1:
        iterator = pd.Series(unfinished_results.loc[[dataset_name]].index.to_list())
        iterator.apply(lambda x: Utils.safe_run_iteration(idx= x, results= results, Xs=Xs, y=y, n_clusters=n_clusters,
                                                     algorithms=algorithms, random_state=random_state, folder_results=folder_results,
                                                     logs_file=logs_file, error_file=error_file))
    else:
        unfinished_results_dataset = unfinished_results.loc[[dataset_name]]
        unfinished_results_dataset_idx = unfinished_results_dataset.xs(0, level="missing_percentage", drop_level=False).index
        iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns= list(indexes_results.keys()))
        iterator.parallel_apply(lambda x: Utils.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                            n_clusters=n_clusters,
                                                                            algorithms=algorithms,
                                                                            random_state=random_state,
                                                                            folder_results=folder_results,
                                                                            logs_file=logs_file,
                                                                            error_file=error_file), axis= 1)
        unfinished_results_dataset_idx = unfinished_results_dataset.drop(unfinished_results_dataset_idx).index
        iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns= list(indexes_results.keys()))
        iterator.parallel_apply(lambda x: Utils.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                            n_clusters=n_clusters,
                                                                            algorithms=algorithms,
                                                                            random_state=random_state,
                                                                            folder_results=folder_results,
                                                                            logs_file=logs_file,
                                                                            error_file=error_file), axis= 1)
