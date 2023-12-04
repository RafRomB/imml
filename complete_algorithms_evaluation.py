import os.path
import time
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
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

from utils import save_record, run_iteration

folder_name = "results"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_name, filelame)
logs_file = os.path.join(folder_name, 'logs.txt')
error_file = os.path.join(folder_name, 'error.txt')

random_state = 42

parser = argparse.ArgumentParser()
parser.add_argument('start_benchmarking', default= False, type= bool)
args = parser.parse_args()

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


if args.start_benchmarking:
    results = pd.DataFrame(datasets, columns= ["dataset"])
    for k,v in {k:v for k,v in indexes_results.items() if k != "dataset"}.items():
        results = results.merge(pd.Series(v, name= k), how= "cross")
    results = results.set_index(list(indexes_results.keys()))
    results[["finished", "completed"]] = False
else:
    results = pd.read_csv(file_path, index_col= list(indexes_results.keys()))
    results_ = results.select_dtypes(object).drop(columns= ["comments", "stratified"]).replace(np.nan, "np.nan")
    for col in results_.columns:
        results[col] = results_[col].apply(eval)
        
    open(logs_file, 'w').close()
    open(error_file, 'w').close()

unfinished_results = results.loc[~results["finished"]]

for dataset_name in unfinished_results.index.get_level_values("dataset").unique():
    names = dataset_name.split("_")
    x_name,y_name = names if len(names) >1 else (names[0],"0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle= False)
    y = y[y_name]
    n_clusters = y.nunique()

    iterator = pd.Series(unfinished_results.loc[unfinished_results.index.get_level_values("dataset") == dataset_name].index.to_list())
    iterator.apply(lambda x: run_iteration(idx= x, results= results, Xs=Xs, y=y, n_clusters=n_clusters,
                                           algorithms=algorithms, random_state=random_state, file_path=file_path,
                                           logs_file=logs_file, error_file=error_file))
