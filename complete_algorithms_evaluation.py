import itertools
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

from utils import Utils


folder_results = "results"
folder_subresults = "subresults"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_results, filelame)
subresults_path = os.path.join(folder_results, folder_subresults)
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
amputation_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR']
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
indexes_results = {"dataset": datasets, "algorithm": list(algorithms.keys()), "missing_percentage": probs,
                   "amputation_mechanism": amputation_mechanisms, "imputation": imputation, "run_n": runs_per_alg}
indexes_names = list(indexes_results.keys())

if not args.continue_benchmarking:
    if not eval(input("Are you sure you want to start benchmarking and delete previous results? (True/False)")):
        raise Exception
    results = pd.DataFrame(datasets, columns= ["dataset"])
    for k,v in {k:v for k,v in indexes_results.items() if k != "dataset"}.items():
        results = results.merge(pd.Series(v, name= k), how= "cross")
    results.loc[(results["amputation_mechanism"] == "EDM") & (
                results["missing_percentage"] == 0), "amputation_mechanism"] = "'None'"
    results = results.set_index(indexes_names)

    idx_to_drop = results.xs(0, level="missing_percentage",
                             drop_level=False).xs(True, level="imputation", drop_level=False).index
    results = results.drop(idx_to_drop)
    for amputation_mechanism in amputation_mechanisms[1:]:
        idx_to_drop = results.xs(0, level="missing_percentage",
                                 drop_level=False).xs(amputation_mechanism, level="amputation_mechanism", drop_level=False).index
        results = results.drop(idx_to_drop)
    results_amputation_mechanism_none = results.xs(0, level="missing_percentage", drop_level=False)
    results_amputation_mechanism_none_tochange = results_amputation_mechanism_none.index.to_frame()
    results_amputation_mechanism_none_tochange["amputation_mechanism"] = "None"
    results.loc[results_amputation_mechanism_none.index].index = pd.MultiIndex.from_frame(results_amputation_mechanism_none_tochange)

    for amputation_mechanism, dataset in itertools.product(["MAR", "MNAR"], ["nutrimouse_genotype",
                                                                             "nutrimouse_diet",
                                                                             "metabric",
                                                                             "bdgp",
                                                                             "buaa",
                                                                             # "simulated_netMUG",
                                                                             ]):
        idx_to_drop = results.xs(dataset, level="dataset",
                                 drop_level=False).xs(amputation_mechanism, level="amputation_mechanism", drop_level=False).index
        results = results.drop(idx_to_drop)

    results[["finished", "completed"]] = False
    results.to_csv(file_path)

    shutil.rmtree(subresults_path, ignore_errors=True)
    os.mkdir(subresults_path)

    os.remove(logs_file) if os.path.exists(logs_file) else None
    os.remove(error_file) if os.path.exists(error_file) else None
    open(logs_file, 'w').close()
    open(error_file, 'w').close()
else:
    results = pd.read_csv(file_path, index_col= indexes_names)
    results = Utils.collect_subresults(results=results, subresults_path=subresults_path, indexes_names=indexes_names)

unfinished_results = results.loc[~results["finished"]]

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
        iterator.apply(lambda x: Utils.run_iteration(idx= x, results= results, Xs=Xs, y=y, n_clusters=n_clusters,
                                                     algorithms=algorithms, random_state=random_state, subresults_path=subresults_path,
                                                     logs_file=logs_file, error_file=error_file), axis= 1)
    else:
        try:
            unfinished_results_dataset_idx = unfinished_results_dataset.xs(0, level="missing_percentage", drop_level=False).index
            iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns= indexes_names)
            iterator.parallel_apply(lambda x: Utils.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                  n_clusters=n_clusters,
                                                                  algorithms=algorithms,
                                                                  random_state=random_state,
                                                                  subresults_path=subresults_path,
                                                                  logs_file=logs_file,
                                                                  error_file=error_file), axis= 1)
            results = Utils.collect_subresults(results=results, subresults_path=subresults_path,
                                               indexes_names=indexes_names)
            results.to_csv(file_path)

            unfinished_results_dataset_idx = unfinished_results_dataset.drop(unfinished_results_dataset_idx).index
            iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns=indexes_names)
        except KeyError:
            iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)

        iterator.parallel_apply(lambda x: Utils.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                            n_clusters=n_clusters,
                                                                            algorithms=algorithms,
                                                                            random_state=random_state,
                                                                            subresults_path=subresults_path,
                                                                            logs_file=logs_file,
                                                                            error_file=error_file), axis= 1)
        results = Utils.collect_subresults(results=results, subresults_path=subresults_path,
                                           indexes_names=indexes_names)
        results.to_csv(file_path)

