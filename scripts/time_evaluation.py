import os
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans
# from mvlearn.decomposition import AJIVE, GroupPCA
# from mvlearn.cluster import MultiviewSpectralClustering, MultiviewCoRegSpectralClustering
from imvc.cluster import DAIMC, EEIMVC, IMSR, LFIMVC, MKKMIK, MSNE, OSLFIMVC, SIMCADC
from imvc.datasets import LoadDataset
from imvc.decomposition import DeepMF, DFMF, MOFA
from imvc.transformers import MultiViewTransformer, ConcatenateViews, NormalizerNaN
from imvc.algorithms import NMFC

from models import Model
from settings import TIME_RESULTS_PATH, TIME_LOGS_PATH, TIME_ERRORS_PATH, RANDOM_STATE

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
    # "Concat": {"alg": make_pipeline(ConcatenateViews(),
    #                                 StandardScaler().set_output(transform='pandas'),
    #                                 KMeans()), "params": {}},
    # "NMFC": {"alg": make_pipeline(ConcatenateViews(),
    #                               MinMaxScaler().set_output(transform='pandas'),
    #                               NMFC().set_output(transform='pandas')), "params": {}},
    # "MVSpectralClustering": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
    #                                               MultiviewSpectralClustering()),
    #                          "params": {}},
    # "MVCoRegSpectralClustering": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
    #                                                    MultiviewCoRegSpectralClustering()),
    #                               "params": {}},
    # "GroupPCA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), GroupPCA(), StandardScaler(), KMeans()),
    #              "params": {}},
    # "AJIVE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), AJIVE(),
    #                                MultiViewTransformer(FunctionTransformer(pd.DataFrame)), ConcatenateViews(),
    #                                StandardScaler(), KMeans()),
    #           "params": {}},
    # "SNF": {"alg": MultiViewTransformer(StandardScaler().set_output(transform="pandas")), "params": {}},
    # "DAIMC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
    #                                DAIMC()), "params": {}},
    # "EEIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                                 EEIMVC()), "params": {}},
    # "LFIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                                 LFIMVC()), "params": {}},
    # "MKKMIK": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                                 MKKMIK()), "params": {}},
    # "MSNE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                               MSNE()), "params": {}},
    "OSLFIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                      OSLFIMVC()), "params": {}},
    # "SIMCADC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
    #                                  SIMCADC()), "params": {}},
    # # "DeepMF": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), ConcatenateViews(),
    # #                                 DeepMF(), StandardScaler(), KMeans()),
    # #              "params": {}},
    # "DFMF": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")), DFMF().set_output(transform="pandas"),
    #                               StandardScaler().set_output(transform="pandas"), KMeans()),
    #          "params": {}},
    # "MOFA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")), MOFA().set_output(transform="pandas"),
    #                               ConcatenateViews(), StandardScaler().set_output(transform="pandas"), KMeans()),
    #          "params": {}},
}

if os.path.exists(TIME_RESULTS_PATH):
    results = pd.read_csv(TIME_RESULTS_PATH, index_col=0)
    results_1 = pd.DataFrame(0, index=algorithms.keys(), columns=datasets)
    results = pd.concat([results, results_1.loc[results_1.index.difference(results.index)]])
else:
    results = pd.DataFrame(0, index=algorithms.keys(), columns= datasets)
    results.loc[["MVSpectralClustering", "MVCoRegSpectralClustering", "SNF"], "nuswide"] = np.nan
    os.remove(TIME_LOGS_PATH) if os.path.exists(TIME_LOGS_PATH) else None
    os.remove(TIME_ERRORS_PATH) if os.path.exists(TIME_ERRORS_PATH) else None
    open(TIME_LOGS_PATH, 'w').close()
    open(TIME_ERRORS_PATH, 'w').close()

errors_dict = defaultdict(int)

for dataset_name in datasets:
    names = dataset_name.split("_")
    if "simulated" in names:
        names = ["_".join(names)]
    x_name, y_name = names if len(names) > 1 else (names[0], "0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle=True, random_state=RANDOM_STATE)
    y = y[y_name]
    n_clusters = y.nunique()

    for alg_name, alg in algorithms.items():
        time_execution = results.loc[alg_name, dataset_name]
        if (time_execution > 0) or np.isnan(time_execution):
            continue
        with open(TIME_LOGS_PATH, "a") as f:
            f.write(f'\n {dataset_name} \t {alg_name} \t {datetime.now()}')

        try:
            start_time = time.perf_counter()
            clusters, _ = Model(alg_name=alg_name, alg=alg).method(train_Xs=Xs, n_clusters=n_clusters,
                                                                       random_state=RANDOM_STATE, run_n=0)
            elapsed_time = time.perf_counter() - start_time
        except Exception as exception:
            errors_dict[f"{type(exception).__name__}: {exception}"] += 1
            with open(TIME_ERRORS_PATH, "a") as f:
                f.write(f'\n {dataset_name} \t {alg_name} \t {errors_dict} \t {datetime.now()}')
            elapsed_time = np.nan

        results.loc[alg_name, dataset_name] = elapsed_time
        results.to_csv(TIME_RESULTS_PATH)

print("Completed successfully!")
with open(TIME_ERRORS_PATH, "a") as f:
    f.write(f'\n Completed successfully \t {datetime.now()}')

