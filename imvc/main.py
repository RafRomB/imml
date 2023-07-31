import itertools
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from reval.utils import compute_metrics
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from validclust import dunn

from imvc.pipelines import MOFAPipeline, MONETPipeline, MSNEPipeline, SUMOPipeline, NMFCPipeline, ConcatPipeline
from imvc.datasets import LoadDataset
from imvc.utils import DatasetUtils
from imvc.transformers import ConcatenateViews
from utils.utils import BugInMONET

random_state = 0
START_BENCHMARKING = True

Xs, y = LoadDataset.load_incomplete_nutrimouse(p = 0, return_y = True, random_state = random_state)
nutrimouse_genotype = Xs, LabelEncoder().fit_transform(y[0]), y[0].squeeze().nunique(), "nutrimouse_genotype"
nutrimouse_diet = Xs, LabelEncoder().fit_transform(y[1]), y[1].squeeze().nunique(), "nutrimouse_diet"
datasets = [nutrimouse_genotype, nutrimouse_diet]
probs = np.arange(0., 1., step= 0.1).round(1)
algorithms = {
    "Concat": {"alg": ConcatPipeline, "params": {}},
    "MOFA": {"alg": MOFAPipeline, "params": {}},
    "NMFC": {"alg": NMFCPipeline, "params": {}},
    "MONET": {"alg": MONETPipeline, "params": {"n_jobs":-1}},
    "MSNE": {"alg": MSNEPipeline, "params": {"n_jobs":8}},
    "SUMO": {"alg": SUMOPipeline, "params": {}},
}
runs_per_alg = np.arange(10).tolist()

iterations = itertools.product(algorithms.items(), datasets, probs, runs_per_alg)

if START_BENCHMARKING:
    results = pd.DataFrame()
else:
    results = pd.read_csv("results.csv")
    for _ in range(len(results)):
        next(iterations)

for (alg_name, alg_comp), (Xs, y, n_clusters, dataset_name), p, i in iterations:
    alg = alg_comp["alg"]
    params = alg_comp["params"]
    incomplete_Xs = DatasetUtils.convert_mvd_into_imvd(Xs=Xs, p=p, assess_percentage = True,
                                                       random_state = random_state + i)
    print(f"Algorithm: {alg_name} \t Dataset: {dataset_name} \t Missing: {p} \t Iteration: {i}")
    if alg_name == "MONET":
        model = alg(random_state = random_state + i, **params)
    else:
        model = alg(n_clusters=n_clusters, random_state = random_state + i, **params)
    model.estimator = model.estimator.set_params(verbose=False)
    keep_running = True
    errors_dict = defaultdict(int)
    while keep_running:
        try:
            start_time = time.perf_counter()
            clusters = model.fit_predict(incomplete_Xs)
            keep_running = False
        except (NameError, BugInMONET) as exception:
            model.estimator.set_params(random_state=model.estimator.random_state + 1)
            errors_dict[type(exception).__name__] += 1
            if sum(errors_dict.values()) == 100:
                keep_running = False
                continue
    if sum(errors_dict.values()) == 100:
        continue
    elapsed_time = time.perf_counter() - start_time
    if alg_name in ["MOFA", "Concat"]:
        X = make_pipeline(*model.transformers).transform(incomplete_Xs)
    elif alg_name in ["NMFC"]:
        X = model.transform(incomplete_Xs)
    else:
        X = make_pipeline(*model.transformers).transform(incomplete_Xs)
        X = make_pipeline(ConcatenateViews(), SimpleImputer().set_output(transform="pandas")).fit_transform(X)
    #todo test pipeline index when it is a list
    mask = ~np.isnan(clusters)
    labels_pred = np.nan_to_num(clusters, nan = -1).astype(int)
    labels_pred = pd.factorize(labels_pred)[0]
    random_preds = [pd.Series(y).value_counts().index[0]] * len(y)

    supervised_metrics = {
        **compute_metrics(class_labels=y, clust_labels=labels_pred, perm=True),
        "ami": metrics.adjusted_mutual_info_score(labels_true=y, labels_pred=labels_pred),
        "ari": metrics.adjusted_rand_score(labels_true=y, labels_pred=labels_pred),
        "completeness": metrics.completeness_score(labels_true=y, labels_pred=labels_pred),
    }
    if len(np.unique(labels_pred)) == 1:
        unsupervised_metrics = {"silhouette": np.nan, "vrc": np.nan, "db": np.nan, "dunn": np.nan}
    else:
        unsupervised_metrics = {
            "silhouette": metrics.silhouette_score(X = X, labels = labels_pred, random_state= random_state),
            "vrc": metrics.calinski_harabasz_score(X = X, labels = labels_pred),
            "db": metrics.davies_bouldin_score(X = X, labels = labels_pred),
            "dunn": dunn(dist = pairwise_distances(X), labels = labels_pred),
        }

    dict_results = {
        "alg": alg_name,
        "dataset": dataset_name,
        "n_samples": len(X),
        "missing_percentage": int(100*p),
        "time": elapsed_time,
        "execution": i,
        **supervised_metrics,
        **unsupervised_metrics,
        "random_acc": metrics.accuracy_score(y_true=y, y_pred=random_preds),
        "random_f1": metrics.f1_score(y_true=y, y_pred=random_preds, average='macro'),
        "label_sizes": pd.Series(y).value_counts().to_dict(),
        "cluster_sizes": pd.Series(clusters).value_counts(dropna=False).to_dict(),
        "relative_label_sizes": pd.Series(y).value_counts(normalize=True).to_dict(),
        "relative_cluster_sizes": pd.Series(clusters).value_counts(normalize=True, dropna=False).to_dict(),
        "n_clustered_samples": mask.sum(),
        "percentage_clustered_samples": 100*mask.sum() // len(X),
    }

    if alg_name == "MONET":
        X = X.iloc[mask]
        labels_true = y[mask]
        labels_pred = clusters[mask].astype(int)
        labels_pred = pd.factorize(labels_pred)[0]
        sub_random_preds = [pd.Series(labels_true).value_counts().index[0]] * len(labels_true)
        supervised_metrics = {
            **compute_metrics(class_labels=labels_true, clust_labels=labels_pred, perm=True),
            "ami": metrics.adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred),
            "ari": metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred),
            "completeness": metrics.completeness_score(labels_true=labels_true, labels_pred=labels_pred),
        }
        unsupervised_metrics = {
            "silhouette": metrics.silhouette_score(X=X, labels=labels_pred, random_state=random_state),
            "vrc": metrics.calinski_harabasz_score(X=X, labels=labels_pred),
            "db": metrics.davies_bouldin_score(X=X, labels=labels_pred),
            "dunn": dunn(dist=pairwise_distances(X), labels=labels_pred),
        }
        dict_subresults = {
            **supervised_metrics,
            **unsupervised_metrics,
            "random_acc": metrics.accuracy_score(y_true=labels_true, y_pred=sub_random_preds),
            "random_f1": metrics.f1_score(y_true=labels_true, y_pred=sub_random_preds, average='macro'),
            "label_sizes": pd.Series(labels_true).value_counts().to_dict(),
            "relative_label_sizes": pd.Series(labels_true).value_counts(normalize=True).to_dict(),
        }
        dict_subresults = {f"sub_{key}": value for key,value in dict_subresults.items()}
        dict_subresults["comments"] = dict(errors_dict)
        dict_results = dict_results | dict_subresults

    results = pd.concat([results, pd.DataFrame([dict_results])])
    results.to_csv("results.csv", index=False)

