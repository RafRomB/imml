import os
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from bignmf.models.jnmf.integrative import IntegrativeJnmf
from bignmf.models.jnmf.standard import StandardJnmf
from reval.utils import kuhn_munkres_algorithm
from sklearn import metrics
from sklearn.cluster import spectral_clustering
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from snf import compute
from validclust import dunn

from imvc.transformers import MultiViewTransformer, ConcatenateViews, Ampute
from imvc.utils import DatasetUtils


class Utils:


    @staticmethod
    def run_iteration(idx, results, Xs, y, n_clusters, algorithms, random_state, subresults_path, logs_file, error_file):
        errors_dict = defaultdict(int)
        row = results.loc[[idx]]
        try:
            with open(logs_file, "a") as f:
                f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {datetime.now()}')
            row_index = row.index
            alg_name, p, amputation_mechanism, impute, run_n = (
                row_index.get_level_values("algorithm")[0],
                row_index.get_level_values("missing_percentage")[0] / 100,
                row_index.get_level_values("amputation_mechanism")[0],
                row_index.get_level_values("imputation")[0],
                row_index.get_level_values("run_n")[0])

            alg = algorithms[alg_name]
            train_Xs = DatasetUtils.shuffle_imvd(Xs=Xs, random_state=random_state + run_n)
            y_train = y.loc[train_Xs[0].index]
            strat = False
            if p != 0:
                if amputation_mechanism == "ED":
                    try:
                        assert n_clusters < len(train_Xs[0]) * (1-p)
                    except AssertionError as exception:
                        raise AssertionError(f"{exception}; n_clusters < len(train_Xs[0]) * (1-p)")
                    amp = Ampute(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state + run_n,
                                 assess_percentage=True, stratify=y_train)
                    try:
                        train_Xs = amp.fit_transform(train_Xs)
                        strat = True
                    except ValueError:
                        amp = Ampute(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state + run_n,
                                     assess_percentage=True)
                        train_Xs = amp.fit_transform(train_Xs)
                else:
                    amp = Ampute(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state + run_n)
                    train_Xs = amp.fit_transform(train_Xs)

            if impute:
                train_Xs = MultiViewTransformer(SimpleImputer(strategy="mean").set_output(
                    transform="pandas")).fit_transform(train_Xs)
            else:
                train_Xs = DatasetUtils.select_complete_samples(Xs=train_Xs)
                y_train = y_train.loc[train_Xs[0].index]

            start_time = time.perf_counter()
            if alg_name == "SNF":
                preprocessing_step = MultiViewTransformer(StandardScaler().set_output(transform="pandas"))
                train_Xs = preprocessing_step.fit_transform(train_Xs)
                k_snf = np.ceil(len(y_train)/10).astype(int)
                affinities = compute.make_affinity(train_Xs, normalize=False, K= k_snf)
                fused = compute.snf(affinities, K= k_snf)
                clusters = spectral_clustering(fused, n_clusters=n_clusters, random_state=random_state + run_n)
            elif alg_name == "intNMF":
                preprocessing_step = MultiViewTransformer(MinMaxScaler().set_output(transform="pandas"))
                train_Xs = preprocessing_step.fit_transform(train_Xs)
                model = IntegrativeJnmf({k: v for k, v in enumerate(train_Xs)}, k=n_clusters, lamb=0.1)
                model.run(trials=50, iterations=100, verbose=False)
                model.cluster_data()
                clusters = np.argmax(model.w_cluster, axis=1)
            elif alg_name == "jNMF":
                preprocessing_step = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")))
                train_Xs = preprocessing_step.fit_transform(train_Xs)
                model = StandardJnmf({k: v for k, v in enumerate(train_Xs)}, k=n_clusters)
                model.run(trials=50, iterations=100, verbose=False)
                model.cluster_data()
                clusters = np.argmax(model.w_cluster, axis=1)
            else:
                model, params = alg["alg"], alg["params"]
                if alg_name == "GroupPCA":
                    model[1].set_params(n_components=n_clusters, random_state=random_state + run_n, multiview_output=False)
                elif alg_name == "AJIVE":
                    model[1].set_params(random_state=random_state + run_n)
                if alg_name == "NMFC":
                    model[-1].set_params(n_components=n_clusters, random_state=random_state + run_n)
                else:
                    model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
                clusters = model.fit_predict(train_Xs)

            clusters = pd.Series(clusters, index=y_train.index)

            elapsed_time = time.perf_counter() - start_time

            if alg_name in ["NMFC"]:
                train_X = model.transform(train_Xs)
            elif alg_name in ["SNF"]:
                train_X = preprocessing_step.transform(train_Xs)
            elif alg_name in ["intNMF", "jNMF"]:
                train_X = model.w
            else:
                train_X = model[:-1].transform(train_Xs)
            if isinstance(train_X, list):
                train_X = ConcatenateViews().fit_transform(train_X)
            if not isinstance(train_X, pd.DataFrame):
                train_X = pd.DataFrame(train_X, index=y_train.index)

            assert train_X.index.equals(y_train.index)
            assert train_Xs[0].index.equals(y_train.index)

            if p > 0:
                multiidx = []
                for level in row_index.names:
                    if level == "missing_percentage":
                        multiidx_value = 0
                    elif level == "amputation_mechanism":
                        multiidx_value = "EDM"
                    elif level == "imputation":
                        multiidx_value = False
                    else:
                        multiidx_value = row_index.get_level_values(level=level)[0]
                    multiidx.append([multiidx_value])
                best_solution = pd.MultiIndex.from_arrays(multiidx, names=row_index.names)
                best_solution = results.loc[best_solution].iloc[0]
                best_solution = pd.Series(best_solution["y_pred"], index=best_solution["y_pred_idx"])
            else:
                best_solution = None

            dict_results = Utils.save_record(train_Xs=train_Xs, train_X=train_X, clusters=clusters, y=y_train, p=p,
                                       best_solution=best_solution, elapsed_time=elapsed_time, strat=strat,
                                       random_state=random_state, errors_dict=errors_dict)
            dict_results = pd.DataFrame(pd.Series(dict_results), columns=row_index).T
            row[dict_results.columns] = dict_results
            row[["finished", "completed"]] = True
        except Exception as exception:
            errors_dict[f"{type(exception).__name__}: {exception}"] += 1
            row[["finished", "comments"]] = True, dict(errors_dict)
            with open(error_file, "a") as f:
                f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {errors_dict}')
            # raise

        row.to_csv(os.path.join(subresults_path, f"{'_'.join([str(i) for i in idx])}.csv"))
        return row

    @staticmethod
    def save_record(train_Xs, train_X, clusters, y, p, best_solution, elapsed_time, strat, random_state, errors_dict):

        missing_clusters_mask = np.invert(np.isnan(clusters))

        dict_results = {
            "n_samples": len(train_X),
            "n_incomplete_samples": DatasetUtils.get_n_incomplete_samples(train_Xs),
            "n_complete_samples": DatasetUtils.get_n_complete_samples(train_Xs),
            "time": elapsed_time,
            "n_clustered_samples": missing_clusters_mask.sum(),
            "percentage_clustered_samples": 100* missing_clusters_mask.sum() // len(train_X),
            "comments": dict(errors_dict),
            "stratified": strat,
            "missing_view_profile": DatasetUtils.get_missing_view_profile(Xs=train_Xs).values.tolist(),
        }

        if not all(missing_clusters_mask):
            clusters_excluding_missing, X_excluding_missing, y_excluding_missing = (clusters[missing_clusters_mask].astype(int),
                                                                                    train_X.iloc[missing_clusters_mask],
                                                                                    y[missing_clusters_mask])
            clusters_excluding_missing = pd.factorize(clusters_excluding_missing)[0]
            summary_results = Utils.get_result_summary(y_true=y_excluding_missing, y_pred=clusters_excluding_missing,
                                                 p=p, best_solution=best_solution,
                                                 X=X_excluding_missing, random_state=random_state)
            summary_results = {f"{key}_excluding_clusters_missing": value for key, value in summary_results.items()}
            dict_results = {**dict_results, **summary_results}
            clusters[~missing_clusters_mask] = -1
        summary_results = Utils.get_result_summary(y_true=y, y_pred=clusters, p=p, best_solution=best_solution,
                                                   X=train_X, random_state=random_state)
        dict_results = {**dict_results, **summary_results}

        return dict_results


    @staticmethod
    def get_result_summary(y_true, y_pred, p, best_solution, X, random_state):
        assert y_true.index.equals(y_pred.index)
        supervised_metrics = Utils.compute_supervised_metrics(y_true=y_true, y_pred=y_pred)
        if p > 0:
            best_solution_local = best_solution.loc[y_pred.index]
            assert best_solution_local.index.equals(y_pred.index)
            performance_metrics = Utils.compute_supervised_metrics(y_true=best_solution_local, y_pred=y_pred)
            performance_metrics = {f"{key}_performance": value for key, value in performance_metrics.items()}

            supervised_metrics = {**supervised_metrics, **performance_metrics}

        unsupervised_metrics = Utils.compute_unsupervised_metrics(X=X, y_pred=y_pred, random_state=random_state)
        summary_dict = {
            "label_sizes": pd.Series(y_true).value_counts().to_dict(),
            "cluster_sizes": pd.Series(y_pred).value_counts(dropna=False).to_dict(),
            "relative_label_sizes": pd.Series(y_true).value_counts(normalize=True).to_dict(),
            "relative_cluster_sizes": pd.Series(y_pred).value_counts(normalize=True, dropna=False).to_dict(),
            "y_true": y_true.to_list(),
            "y_pred": y_pred.to_list(),
            "y_true_idx": y_true.index.to_list(),
            "y_pred_idx": y_pred.index.to_list(),
            **supervised_metrics,
            **unsupervised_metrics,
        }
        return summary_dict


    @staticmethod
    def compute_supervised_metrics(y_true, y_pred):
        random_preds = [pd.Series(y_true).value_counts().index[0]] * len(y_true)
        supervised_metrics = {
            **Utils.get_supervised_metrics(y_true=y_true, y_pred=y_pred),
            "bal_acc": metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
            "ami": metrics.adjusted_mutual_info_score(labels_true=y_true, labels_pred=y_pred),
            "ari": metrics.adjusted_rand_score(labels_true=y_true, labels_pred=y_pred),
            "completeness": metrics.completeness_score(labels_true=y_true, labels_pred=y_pred),
            "random_acc": metrics.accuracy_score(y_true=y_true, y_pred=random_preds),
            "random_f1": metrics.f1_score(y_true=y_true, y_pred=random_preds, average='macro'),
        }
        return supervised_metrics


    @staticmethod
    def compute_unsupervised_metrics(X, y_pred, random_state):
        if len(np.unique(y_pred)) == 1:
            unsupervised_metrics = {"silhouette": np.nan, "vrc": np.nan, "db": np.nan, "dunn": np.nan}
        else:
            unsupervised_metrics = {
                "silhouette": metrics.silhouette_score(X = X, labels = y_pred, random_state= random_state),
                "vrc": metrics.calinski_harabasz_score(X = X, labels = y_pred),
                "db": metrics.davies_bouldin_score(X = X, labels = y_pred),
                "dunn": dunn(dist = metrics.pairwise_distances(X), labels = y_pred),
            }
        return unsupervised_metrics


    @staticmethod
    def get_supervised_metrics(y_true, y_pred):
        perm_clust_labels = kuhn_munkres_algorithm(true_lab=y_true, pred_lab=y_pred)
        scores = {
            'ACC': metrics.accuracy_score(y_true=y_true, y_pred=perm_clust_labels),
            'MCC': metrics.matthews_corrcoef(y_true=y_true, y_pred=perm_clust_labels),
            'F1': metrics.f1_score(y_true=y_true, y_pred=perm_clust_labels, average='macro'),
            'precision': metrics.precision_score(y_true=y_true, y_pred=perm_clust_labels, average='macro', zero_division=0),
            'recall': metrics.recall_score(y_true=y_true, y_pred=perm_clust_labels, average='macro', zero_division=0)}
        return scores


    @staticmethod
    def collect_subresults(results, subresults_path, indexes_names):
        subresults_files = pd.Series(os.listdir(subresults_path)).apply(lambda x: os.path.join(subresults_path, x))
        subresults_files = subresults_files[subresults_files.apply(os.path.isfile)]
        subresults_files = pd.concat(subresults_files.apply(pd.read_csv).to_list())
        subresults_files = subresults_files.set_index(indexes_names)
        results.loc[subresults_files.index, subresults_files.columns] = subresults_files
        drop_columns = ["comments", "stratified"] if "stratified" in results.select_dtypes(object).columns else "comments"
        results_ = results.select_dtypes(object).drop(columns=drop_columns).replace(np.nan, "np.nan")
        for col in results_.columns:
            results[col] = results_[col].apply(eval)
        return results


