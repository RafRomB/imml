import numpy as np
import pandas as pd
from reval.utils import compute_metrics
from sklearn import metrics
from validclust import dunn


def save_record(labels_pred, y, X, random_state, alg_name, dataset_name, p, elapsed_time, i, random_preds, clusters,
                mask, errors_dict):
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
            "dunn": dunn(dist = metrics.pairwise_distances(X), labels = labels_pred),
        }

    dict_results = {
        "alg": alg_name,
        "dataset": dataset_name,
        "n_samples": len(X),
        "% incomplete samples": int(100*p),
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
            "dunn": dunn(dist=metrics.pairwise_distances(X), labels=labels_pred),
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

    return dict_results
