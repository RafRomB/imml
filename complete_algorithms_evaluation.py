import itertools
import os.path
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from imvc.pipelines import NMFCPipeline, ConcatPipeline
from imvc.datasets import LoadDataset
from imvc.utils import DatasetUtils
from imvc.transformers import SelectCompleteSamples
from utils.utils import save_record

random_state = 0
START_BENCHMARKING = True
folder_name = "results"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_name, filelame)

Xs, y = LoadDataset.load_incomplete_nutrimouse(p = 0, return_y = True, random_state = random_state)
nutrimouse_genotype = Xs, LabelEncoder().fit_transform(y[0]), y[0].squeeze().nunique(), "nutrimouse_genotype"
nutrimouse_diet = Xs, LabelEncoder().fit_transform(y[1]), y[1].squeeze().nunique(), "nutrimouse_diet"
datasets = [nutrimouse_genotype, nutrimouse_diet]
probs = np.arange(0., 1., step= 0.1).round(1)
algorithms = {
    "Concat": {"alg": ConcatPipeline, "params": {}},
    "NMFC": {"alg": NMFCPipeline, "params": {}},
}
runs_per_alg = np.arange(10).tolist()
only_complete_samples = [True, False]

iterations = itertools.product(algorithms.items(), datasets, probs, only_complete_samples, runs_per_alg)

if START_BENCHMARKING:
    results = pd.DataFrame()
else:
    results = pd.read_csv(file_path)

for (alg_name, alg_comp), (Xs, y, n_clusters, dataset_name), p, complete_samples, i in iterations:
    alg = alg_comp["alg"]
    params = alg_comp["params"]

    if not START_BENCHMARKING:
        checking_results = ((results["alg"] == alg_name) & (results["dataset"] == dataset_name)
                            & (results["only_complete_samples"] == complete_samples)
                            & (results["missing_percentage"] == int(100*p)) & (results["execution"] == i))
        if not results[checking_results].empty:
            continue

    incomplete_Xs = DatasetUtils.convert_mvd_into_imvd(Xs=Xs, p=p, assess_percentage = True,
                                                       random_state = random_state + i)
    if complete_samples:
        local_y = y[DatasetUtils.get_missing_view_panel(Xs= incomplete_Xs).all(axis= 1)]
        if len(local_y) <= n_clusters:
            continue
        incomplete_Xs = SelectCompleteSamples().transform(incomplete_Xs)
    print(f"Algorithm: {alg_name} \t Dataset: {dataset_name} \t Missing: {p} \t Complete: {complete_samples} \t Iteration: {i}")
    model = alg(n_clusters=n_clusters, random_state = random_state + i, **params)
    start_time = time.perf_counter()
    clusters = model.fit_predict(incomplete_Xs)
    elapsed_time = time.perf_counter() - start_time
    if alg_name in ["Concat"]:
        X = make_pipeline(*model.transformers).transform(incomplete_Xs)
    elif alg_name in ["NMFC"]:
        X = model.transform(incomplete_Xs)

    if complete_samples:
        clusters_2 = pd.Series(np.nan, index= DatasetUtils.get_sample_names(Xs=Xs))
        clusters_2.loc[X.index] = clusters
        clusters = clusters_2.values
        X_2 = pd.DataFrame(np.nan, index= DatasetUtils.get_sample_names(Xs=Xs), columns= X.columns)
        X_2.loc[X.index] = X
        X = SimpleImputer(fill_value="mean").set_output(transform= "pandas").fit_transform(X_2)
    mask = ~np.isnan(clusters)
    labels_pred = np.nan_to_num(clusters, nan = -1).astype(int)
    labels_pred = pd.factorize(labels_pred)[0]
    random_preds = [pd.Series(y).value_counts().index[0]] * len(y)

    dict_results = save_record(labels_pred=labels_pred, y=y, X=X, random_state=random_state, alg_name=alg_name,
                               dataset_name=dataset_name, p=p, elapsed_time=elapsed_time, i=i, random_preds=random_preds,
                               clusters=clusters, mask=mask, errors_dict={})
    dict_results["only_complete_samples"] = complete_samples

    results = pd.concat([results, pd.DataFrame([dict_results])])
    results.to_csv(file_path, index=False)

