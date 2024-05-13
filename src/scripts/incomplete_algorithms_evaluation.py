import itertools
import os.path
import argparse
import shutil
from datetime import datetime

import numpy as np
from pandarallel import pandarallel
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imvc.datasets import LoadDataset
from imvc.transformers import MultiViewTransformer
from imvc.cluster import NEMO, DAIMC
from settings import INCOMPLETE_RESULTS_PATH, SUBRESULTS_PATH, INCOMPLETE_LOGS_PATH, INCOMPLETE_ERRORS_PATH, \
    RANDOM_STATE, TIME_LIMIT, TIME_RESULTS_PATH
from src.utils.create_result_table import CreateResultTable
from src.utils.run_clustering import RunClustering

# args = lambda: None
# args.continue_benchmarking, args.n_jobs, args.save_results = True, 2, False
parser = argparse.ArgumentParser()
parser.add_argument('-continue_benchmarking', default= False, action='store_true')
parser.add_argument('-n_jobs', default= 1, type= int)
parser.add_argument('-save_results', default= False, action='store_true')
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
amputation_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR', "PM"]
probs = np.arange(100, step= 10)
imputation = [True, False]
runs_per_alg = np.arange(25)
algorithms = {
    "NEMO": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                  NEMO()),
                                  "params": {}},
    "DAIMC": {"alg": make_pipeline(
        MultiViewTransformer(FunctionTransformer(lambda x: x.divide(x.pow(2).sum(axis=1).pow(1/2), axis= 0))),
        DAIMC()),
        "params": {}},
}
incomplete_algorithms = True
indexes_results = {"dataset": datasets, "algorithm": list(algorithms.keys()), "missing_percentage": probs,
                   "amputation_mechanism": amputation_mechanisms, "imputation": imputation, "run_n": runs_per_alg}
indexes_names = list(indexes_results.keys())
results = CreateResultTable.create_results_table(datasets=datasets, indexes_results=indexes_results,
                                         indexes_names=indexes_names, amputation_mechanisms=amputation_mechanisms,
                                         two_view_datasets=two_view_datasets)

if not args.continue_benchmarking:
    if not eval(input("Are you sure you want to start benchmarking and delete previous results? (True/False)")):
        raise Exception
    if args.save_results:
        results.to_csv(INCOMPLETE_RESULTS_PATH)

    shutil.rmtree(SUBRESULTS_PATH, ignore_errors=True)
    os.mkdir(SUBRESULTS_PATH)

    os.remove(INCOMPLETE_LOGS_PATH) if os.path.exists(INCOMPLETE_LOGS_PATH) else None
    os.remove(INCOMPLETE_ERRORS_PATH) if os.path.exists(INCOMPLETE_ERRORS_PATH) else None
    open(INCOMPLETE_LOGS_PATH, 'w').close()
    open(INCOMPLETE_ERRORS_PATH, 'w').close()

    results["Run"] = True
    time_results = pd.read_csv(TIME_RESULTS_PATH, index_col=0)

    for dataset_name, (alg_name, alg) in itertools.product(datasets, algorithms.items()):
        if time_results.loc[dataset_name, alg_name] > TIME_LIMIT:
            results.loc[(dataset_name, alg_name), "Run"] = False
else:
    finished_results = pd.read_csv(INCOMPLETE_RESULTS_PATH, index_col= indexes_names)
    results.loc[finished_results.index, finished_results.columns] = finished_results
    finished_results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=SUBRESULTS_PATH,
                                                    indexes_names=indexes_names)
    results.loc[finished_results.index, finished_results.columns] = finished_results

results = results.xs(False, level="imputation", drop_level=False)
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
    'NEMO',
    'DAIMC'
]
unfinished_results = unfinished_results.loc[(datasets_to_run, algorithms_to_run), :]

for dataset_name in unfinished_results.index.get_level_values("dataset").unique():
    names = dataset_name.split("_")
    if "simulated" in names:
        names = ["_".join(names)]
    x_name,y_name = names if len(names) > 1 else (names[0], "0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True)
    y = y[y_name]
    n_clusters = y.nunique()
    unfinished_results_dataset = unfinished_results.loc[[dataset_name]]

    if args.n_jobs == 1:
        iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)
        iterator.apply(lambda x: RunClustering.run_iteration(idx= x, results= results, Xs=Xs, y=y, n_clusters=n_clusters,
                                                         algorithms=algorithms,
                                                         incomplete_algorithms=incomplete_algorithms,
                                                         random_state=RANDOM_STATE,
                                                         subresults_path=SUBRESULTS_PATH, logs_file=INCOMPLETE_LOGS_PATH,
                                                         error_file=INCOMPLETE_ERRORS_PATH), axis= 1)
    else:
        if 0 in unfinished_results_dataset.index.get_level_values("missing_percentage"):
            unfinished_results_dataset_idx = unfinished_results_dataset.xs(0, level="missing_percentage",
                                                                           drop_level=False).index
            iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns= indexes_names)
            iterator.parallel_apply(lambda x: RunClustering.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                      n_clusters=n_clusters,
                                                                      algorithms=algorithms,
                                                                      incomplete_algorithms=incomplete_algorithms,
                                                                      random_state=RANDOM_STATE,
                                                                      subresults_path=SUBRESULTS_PATH,
                                                                      logs_file=INCOMPLETE_LOGS_PATH,
                                                                      error_file=INCOMPLETE_ERRORS_PATH), axis= 1)
            results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=SUBRESULTS_PATH,
                                                   indexes_names=indexes_names)

            if args.save_results:
                results.to_csv(INCOMPLETE_RESULTS_PATH)

            unfinished_results_dataset_idx = unfinished_results_dataset.drop(unfinished_results_dataset_idx).index
            iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns=indexes_names)
        else:
            iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)

        iterator.parallel_apply(lambda x: RunClustering.run_iteration(idx= x, results= results, Xs=Xs, y=y,
                                                                  n_clusters=n_clusters,
                                                                  algorithms=algorithms,
                                                                  incomplete_algorithms=incomplete_algorithms,
                                                                  random_state=RANDOM_STATE,
                                                                  subresults_path=SUBRESULTS_PATH,
                                                                  logs_file=INCOMPLETE_LOGS_PATH,
                                                                  error_file=INCOMPLETE_ERRORS_PATH), axis= 1)
        results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=SUBRESULTS_PATH,
                                               indexes_names=indexes_names)
        if args.save_results:
            results.to_csv(INCOMPLETE_RESULTS_PATH)
        shutil.rmtree(SUBRESULTS_PATH)
        os.mkdir(SUBRESULTS_PATH)

print("Completed successfully!")
with open(INCOMPLETE_LOGS_PATH, "a") as f:
    f.write(f'\n Completed successfully \t {datetime.now()}')
