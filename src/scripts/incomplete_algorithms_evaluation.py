import itertools
import os.path
import argparse
import shutil
from datetime import datetime

import numpy as np
from pandarallel import pandarallel
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imvc.datasets import LoadDataset
from imvc.decomposition import DFMF, MOFA
from imvc.preprocessing import MultiViewTransformer, ConcatenateViews, NormalizerNaN
from imvc.cluster import NEMO, DAIMC, PIMVC, SIMCADC, OSLFIMVC, MSNE, MKKMIK, LFIMVC, EEIMVC
from settings import INCOMPLETE_RESULTS_PATH, SUBRESULTS_PATH, INCOMPLETE_LOGS_PATH, INCOMPLETE_ERRORS_PATH, \
    RANDOM_STATE, TIME_LIMIT, TIME_RESULTS_PATH, DATASET_TABLE_PATH
from src.utils.create_result_table import CreateResultTable
from src.utils.run_clustering import RunClustering

# args = lambda: None
# args.continue_benchmarking, args.n_jobs, args.save_results = True, 2, False
parser = argparse.ArgumentParser()
parser.add_argument('-continue_benchmarking', default= False, action='store_true')
parser.add_argument('-n_jobs', default= 1, type= int)
parser.add_argument('-save_results', default= False, action='store_true')
args = parser.parse_args()

if not args.save_results:
    INCOMPLETE_RESULTS_PATH = os.path.join("test", "incomplete_algorithms_evaluation.csv")
    INCOMPLETE_LOGS_PATH = os.path.join("test", "incomplete_logs.txt")
    INCOMPLETE_ERRORS_PATH = os.path.join("test", "incomplete_errors.txt")


if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

dataset_table = pd.read_csv(DATASET_TABLE_PATH)
dataset_table = dataset_table.reindex(dataset_table.index.append(dataset_table.index[dataset_table["dataset"]=="nutrimouse"])).sort_index().reset_index(drop=True)
dataset_table.loc[dataset_table["dataset"] == "nutrimouse", "dataset"] = ["nutrimouse_genotype", "nutrimouse_diet"]
datasets = dataset_table["dataset"].to_list()
two_view_datasets = dataset_table[dataset_table["n_features"].apply(lambda x: len(eval(x)) == 2)]["dataset"].to_list()

amputation_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR', "PM"]
probs = np.arange(100, step= 10)
imputation = [True, False]
runs_per_alg = np.arange(50)
algorithms = {
    "DAIMC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   DAIMC()), "params": {}},
    "EEIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    EEIMVC()), "params": {}},
    "LFIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    LFIMVC()), "params": {}},
    "MKKMIK": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    MKKMIK()), "params": {}},
    "MSNE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  MSNE()), "params": {}},
    "OSLFIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                      OSLFIMVC()), "params": {}},
    "SIMCADC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                     SIMCADC()), "params": {}},
    "PIMVC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   PIMVC()), "params": {}},
    # "DeepMF": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), ConcatenateViews(),
    #                                 DeepMF(), StandardScaler(), KMeans()),
    #              "params": {}},
    "DFMF": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")), DFMF().set_output(transform="pandas"),
                                  StandardScaler().set_output(transform="pandas"), KMeans()),
             "params": {}},
    "MOFA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")), MOFA().set_output(transform="pandas"),
                                  ConcatenateViews(), StandardScaler().set_output(transform="pandas"), KMeans()),
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

else:
    finished_results = pd.read_csv(INCOMPLETE_RESULTS_PATH, index_col= indexes_names)
    results.loc[finished_results.index, finished_results.columns] = finished_results
    finished_results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=SUBRESULTS_PATH,
                                                    indexes_names=indexes_names)
    results.loc[finished_results.index, finished_results.columns] = finished_results

results["Run"] = True
time_results = pd.read_csv(TIME_RESULTS_PATH, index_col=0)
for dataset_name, (alg_name, alg) in itertools.product(datasets, algorithms.items()):
    time_alg_dat = time_results.loc[dataset_name, alg_name]
    if (time_alg_dat > TIME_LIMIT) and (time_alg_dat <= 0):
        results.loc[(dataset_name, alg_name), "Run"] = False

results = results.xs(False, level="imputation", drop_level=False)
results = results.sort_index(level= "missing_percentage", sort_remaining= False)
unfinished_results = results.loc[~results["finished"]]

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
