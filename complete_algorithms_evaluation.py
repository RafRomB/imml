import os.path
import argparse
import shutil
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

from utils.getresult import GetResult


folder_results = "results"
folder_subresults = "subresults"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_results, filelame)
subresults_path = os.path.join(folder_results, folder_subresults)
logs_file = os.path.join(folder_results, 'logs.txt')
error_file = os.path.join(folder_results, 'errors.txt')

random_state = 42

# args = lambda: None
# args.continue_benchmarking, args.n_jobs = True, 2
parser = argparse.ArgumentParser()
parser.add_argument('-continue_benchmarking', default= False, action='store_true')
parser.add_argument('-n_jobs', default= 1, type= int)
args = parser.parse_args()

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

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
two_view_datasets = ["simulated_gm", "nutrimouse_genotype", "nutrimouse_diet", "metabric", "bdgp",
                     "buaa", "simulated_netMUG"]
amputation_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR']
probs = np.arange(100, step= 10)
imputation = [True, False]
runs_per_alg = np.arange(25)
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
indexes_results = {"dataset": datasets, "algorithm": list(algorithms.keys()), "missing_percentage": probs,
                   "amputation_mechanism": amputation_mechanisms, "imputation": imputation, "run_n": runs_per_alg}
indexes_names = list(indexes_results.keys())
results = GetResult.create_results_table(datasets=datasets, indexes_results=indexes_results,
                                         indexes_names=indexes_names, amputation_mechanisms=amputation_mechanisms,
                                         two_view_datasets=two_view_datasets)

if not args.continue_benchmarking:
    if not eval(input("Are you sure you want to start benchmarking and delete previous results? (True/False)")):
        raise Exception
    results.to_csv(file_path)

    shutil.rmtree(subresults_path, ignore_errors=True)
    os.mkdir(subresults_path)

    os.remove(logs_file) if os.path.exists(logs_file) else None
    os.remove(error_file) if os.path.exists(error_file) else None
    open(logs_file, 'w').close()
    open(error_file, 'w').close()
else:
    finished_results = pd.read_csv(file_path, index_col= indexes_names)
    results.loc[finished_results.index, finished_results.columns] = finished_results
    finished_results = GetResult.collect_subresults(results=results.copy(), subresults_path=subresults_path,
                                                    indexes_names=indexes_names)
    results.loc[finished_results.index, finished_results.columns] = finished_results

results = results.sort_index(level= "missing_percentage", sort_remaining= False)
unfinished_results = results.loc[~results["finished"]]

datasets_to_run = [
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
algorithms_to_run = [
    'Concat',
    'NMFC',
    'MVSpectralClustering',
    'MVCoRegSpectralClustering',
    'GroupPCA',
    'AJIVE',
    'SNF',
    'IntNMF',
    'COCA'
]
unfinished_results = unfinished_results.loc[(datasets_to_run, algorithms_to_run), :]

for dataset_name in unfinished_results.index.get_level_values("dataset").unique():
    names = dataset_name.split("_")
    if "simulated" in names:
        names = ["_".join(names)]
    x_name,y_name = names if len(names) > 1 else (names[0], "0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle= False)
    y = y[y_name]
    n_clusters = y.nunique()
    unfinished_results_dataset = unfinished_results.loc[[dataset_name]]

    if args.n_jobs == 1:
        iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)
        iterator.apply(lambda x: GetResult.run_iteration(idx= x, results= results, Xs=Xs, y=y, n_clusters=n_clusters,
                                                         algorithms=algorithms, random_state=random_state,
                                                         subresults_path=subresults_path, logs_file=logs_file,
                                                         error_file=error_file), axis= 1)
    else:
        if 0 in unfinished_results_dataset.index.get_level_values("missing_percentage"):
            unfinished_results_dataset_idx = unfinished_results_dataset.xs(0, level="missing_percentage",
                                                                           drop_level=False).index
            iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns= indexes_names)
            iterator.parallel_apply(lambda x: GetResult.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                      n_clusters=n_clusters,
                                                                      algorithms=algorithms,
                                                                      random_state=random_state,
                                                                      subresults_path=subresults_path,
                                                                      logs_file=logs_file,
                                                                      error_file=error_file), axis= 1)
            results = GetResult.collect_subresults(results=results.copy(), subresults_path=subresults_path,
                                                   indexes_names=indexes_names)
            results.to_csv(file_path)

            unfinished_results_dataset_idx = unfinished_results_dataset.drop(unfinished_results_dataset_idx).index
            iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns=indexes_names)
        else:
            iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)

        iterator.parallel_apply(lambda x: GetResult.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                  n_clusters=n_clusters,
                                                                  algorithms=algorithms,
                                                                  random_state=random_state,
                                                                  subresults_path=subresults_path,
                                                                  logs_file=logs_file,
                                                                  error_file=error_file), axis= 1)
        results = GetResult.collect_subresults(results=results.copy(), subresults_path=subresults_path,
                                               indexes_names=indexes_names)
        results.to_csv(file_path)
        GetResult.remove_subresults(results=results, subresults_path=subresults_path)

