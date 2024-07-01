import itertools
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from statsmodels.stats.multitest import multipletests
from validclust import dunn
from reval.utils import kuhn_munkres_algorithm

from src.utils import dbcv


class GetMetrics:


    @staticmethod
    def compute_supervised_metrics(y_true, y_pred, random_state = None, n_permutations=1000):
        random_preds = [pd.Series(y_true).value_counts().index[0]] * len(y_true)
        perm_clust_labels = kuhn_munkres_algorithm(true_lab=y_true, pred_lab=y_pred)
        mcc, p_value = GetMetrics.compute_mcc(y_true=y_true, y_pred=perm_clust_labels, random_state=random_state,
                                              n_permutations=n_permutations)
        supervised_metrics = {
            'ACC': metrics.accuracy_score(y_true=y_true, y_pred=perm_clust_labels),
            'MCC': mcc,
            'MCC (p-value)': p_value,
            'F1': metrics.f1_score(y_true=y_true, y_pred=perm_clust_labels, average='macro'),
            'precision': metrics.precision_score(y_true=y_true, y_pred=perm_clust_labels, average='macro', zero_division=0),
            'recall': metrics.recall_score(y_true=y_true, y_pred=perm_clust_labels, average='macro', zero_division=0),
            "bal_acc": metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
            "ami": metrics.adjusted_mutual_info_score(labels_true=y_true, labels_pred=y_pred),
            "ari": metrics.adjusted_rand_score(labels_true=y_true, labels_pred=y_pred),
            "completeness": metrics.completeness_score(labels_true=y_true, labels_pred=y_pred),
            "random_acc": metrics.accuracy_score(y_true=y_true, y_pred=random_preds),
            "random_f1": metrics.f1_score(y_true=y_true, y_pred=random_preds, average='macro'),
        }
        return supervised_metrics


    @staticmethod
    def compute_unsupervised_metrics(X, y_pred, random_state = None):
        if len(np.unique(y_pred)) == 1:
            unsupervised_metrics = {"silhouette": np.nan, "vrc": np.nan, "db": np.nan, "dunn": np.nan}
        else:
            unsupervised_metrics = {
                "silhouette": metrics.silhouette_score(X = X, labels = y_pred, random_state= random_state),
                "vrc": metrics.calinski_harabasz_score(X = X, labels = y_pred),
                "db": metrics.davies_bouldin_score(X = X, labels = y_pred),
                "dbcv": dbcv(X = X.values, labels = y_pred.values),
                "dunn": dunn(dist = metrics.pairwise_distances(X), labels = y_pred),
            }
        return unsupervised_metrics


    @staticmethod
    def compute_mcc(y_true, y_pred, n_permutations=1000, random_state=None):
        mcc_value = metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        if random_state is not None:
            p_values = [metrics.matthews_corrcoef(y_true=shuffle(y_true, random_state=i+random_state),
                                                  y_pred=shuffle(y_pred, random_state=i))
                        for i in range(n_permutations)]
        else:
            p_values = [metrics.matthews_corrcoef(y_true=shuffle(y_true),
                                                  y_pred=shuffle(y_pred))
                        for i in range(n_permutations)]
        p_values = (pd.Series(p_values) > mcc_value).sum() / len(p_values)
        if p_values == 0:
            p_values = 1/(n_permutations + 1)
        return mcc_value, p_values


    @staticmethod
    def save_cluster_evaluation(results_path: str, metrics_path: str, inmetrics_path: str, random_state = None,
                                n_permutations=1000, verbose=True, nb_workers=10, progress_bar=True):
        pandarallel.initialize(nb_workers=nb_workers, progress_bar=progress_bar)
        results = pd.read_csv(results_path)
        if verbose:
            print("results", results.shape)
        results = results[results["finished"]]
        if verbose:
            print("finished_results", results.shape)
        results = results[results["completed"]]
        if verbose:
            print("completed_results", results.shape)
        results[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]] = results[
            ["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].parallel_applymap(eval)
        assert results["y_true_idx"].eq(results["y_pred_idx"]).all()
        supervised_metrics = results[["y_true", "y_pred"]].parallel_apply(
            lambda row: GetMetrics.compute_supervised_metrics(y_true=row["y_true"], y_pred=row["y_pred"],
                                                              random_state=random_state, n_permutations=n_permutations),
            axis=1)
        results = pd.concat([results, pd.DataFrame(supervised_metrics.to_dict()).T], axis=1)
        if verbose:
            print("metrics_results", results.shape)
        indexes_names = ["dataset", "algorithm", "missing_percentage", "amputation_mechanism", "imputation"]
        results = results[results.select_dtypes(include="float").columns.to_list() + indexes_names].groupby(
            indexes_names, sort=False).agg(["mean", 'std']).reset_index()
        results.columns = results.columns.map('_'.join).str.strip('_')
        results["padj"] = multipletests(results["MCC (p-value)_mean"], method="fdr_bh")[1]
        results["log_padj"] = results["padj"].apply(lambda x: -np.log10(x))
        if verbose:
            print("aggregated_results", results.shape)
        results.to_csv(metrics_path, index=None)
        outputs = [results]

        results = pd.merge(results,
                           pd.DataFrame(itertools.product(results["dataset"].unique(), results["algorithm"].unique()),
                                        columns=["dataset", "algorithm"]), how="right")
        res = OneHotEncoder(sparse_output=False).set_output(transform="pandas").fit_transform(
            results[["dataset", "algorithm"]])
        for col in ["silhouette_mean", "silhouette_std", "MCC_mean", "MCC_std", "MCC (p-value)_mean",
                    "MCC (p-value)_std"]:
            res[col] = results[col]
            results[col] = KNNImputer().set_output(transform="pandas").fit_transform(X=res)[col]
            res = res.drop(columns=col)
        results["padj"] = multipletests(results["MCC (p-value)_mean"], method="fdr_bh")[1]
        results["log_padj"] = results["padj"].apply(lambda x: -np.log10(x))
        results.to_csv(inmetrics_path, index=None)
        outputs.append(results)
        return outputs
