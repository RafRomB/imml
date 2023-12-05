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

from imvc.transformers import MultiViewTransformer, ConcatenateViews
from imvc.utils import DatasetUtils


class Utils:


    @staticmethod
    def run_iteration(idx, results, Xs, y, n_clusters, algorithms, random_state, file_path, logs_file, error_file):
        row = results.loc[[idx]]
        with open(logs_file, "a") as f:
            f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {datetime.now()}')
        row_index = row.index
        alg_name, impute, p, run_n = (
            row_index.get_level_values("algorithm")[0],
            row_index.get_level_values("imputation")[0],
            row_index.get_level_values("missing_percentage")[0] / 100,
            row_index.get_level_values("run_n")[0])

        alg = algorithms[alg_name]
        train_Xs = DatasetUtils.shuffle_imvd(Xs=Xs, random_state=random_state + run_n)
        y_train = y.loc[train_Xs[0].index]
        errors_dict = defaultdict(int)
        if p != 0:
            try:
                assert n_clusters < len(train_Xs[0]) * (1-p)
            except AssertionError as exception:
                errors_dict[f"{type(exception).__name__}: {exception}; n_clusters < len(train_Xs[0]) * (1-p)"] += 1
                results.loc[idx, ["finished", "comments"]] = True, dict(errors_dict)
                results.to_csv(file_path)
                with open(error_file, "a") as f:
                    f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {errors_dict}')
                return results.loc[idx]
            try:
                train_Xs = DatasetUtils.add_random_noise_to_views(Xs=train_Xs, p=round(p, 2),
                                                                  random_state=random_state + run_n,
                                                                  assess_percentage=True, stratify=y_train)
                strat = True
            except ValueError:
                try:
                    train_Xs = DatasetUtils.add_random_noise_to_views(Xs=train_Xs, p=round(p, 2),
                                                                      random_state=random_state + run_n,
                                                                      assess_percentage=True)
                    strat = False
                except Exception as exception:
                    errors_dict[f"{type(exception).__name__}: {exception}"] += 1
                    results.loc[idx, ["finished", "comments"]] = True, dict(errors_dict)
                    with open(error_file, "a") as f:
                        f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {errors_dict}')
                    results.to_csv(file_path)
                    return results.loc[idx]
        else:
            strat = False

        if impute:
            train_Xs = MultiViewTransformer(SimpleImputer(strategy="mean").set_output(transform="pandas")).fit_transform(
                train_Xs)
        else:
            train_Xs = DatasetUtils.select_complete_samples(Xs=train_Xs)
            y_train = y_train.loc[train_Xs[0].index]

        try:
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
        except ValueError as exception:
            errors_dict[f"{type(exception).__name__}: {exception}"] += 1
            results.loc[idx, ["finished", "comments"]] = True, dict(errors_dict)
            results.to_csv(file_path)
            with open(error_file, "a") as f:
                f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {errors_dict}')
            return results.loc[idx]

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
            best_solution = pd.MultiIndex.from_arrays(
                [[row_index.get_level_values(level=level)[0]] if level != "missing_percentage" else [0]
                 for level in row_index.names], names=row_index.names)
            best_solution = results.loc[best_solution].iloc[0]
            y_train_total = pd.Series(best_solution["y_true"], index=best_solution["y_pred_idx"])
            best_solution = pd.Series(best_solution["y_pred"], index=best_solution["y_pred_idx"])
        else:
            best_solution = None
            y_train_total = None

        dict_results = Utils.save_record(train_Xs=train_Xs, train_X=train_X, clusters=clusters, y=y_train, p=p, y_true_total=y_train_total,
                                   best_solution=best_solution, elapsed_time=elapsed_time, strat=strat,
                                   random_state=random_state, errors_dict=errors_dict)
        dict_results = pd.DataFrame(pd.Series(dict_results), columns=row_index).T
        results.loc[[idx], dict_results.columns] = dict_results
        results.loc[idx, ["finished", "completed"]] = True
        results.to_csv(file_path)
        return results.loc[idx]


    @staticmethod
    def save_record(train_Xs, train_X, clusters, y, p, best_solution, y_true_total, elapsed_time, strat, random_state, errors_dict):

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
        }

        if not all(missing_clusters_mask):
            clusters_excluding_missing, X_excluding_missing, y_excluding_missing = (clusters[missing_clusters_mask].astype(int),
                                                                                    train_X.iloc[missing_clusters_mask],
                                                                                    y[missing_clusters_mask])
            clusters_excluding_missing = pd.factorize(clusters_excluding_missing)[0]
            summary_results = Utils.get_result_summary(y_true=y_excluding_missing, y_pred=clusters_excluding_missing,
                                                 p=p, best_solution=best_solution, y_true_total=y_true_total,
                                                 X=X_excluding_missing, random_state=random_state)
            summary_results = {f"{key}_excluding_clusters_missing": value for key, value in summary_results.items()}
            dict_results = {**dict_results, **summary_results}
            clusters[~missing_clusters_mask] = -1
        summary_results = Utils.get_result_summary(y_true=y, y_pred=clusters, p=p, best_solution=best_solution,
                                             y_true_total=y_true_total, X=train_X, random_state=random_state)
        dict_results = {**dict_results, **summary_results}

        return dict_results


    @staticmethod
    def get_result_summary(y_true, y_pred, p, best_solution, y_true_total, X, random_state):
        assert y_true.index.equals(y_pred.index)
        supervised_metrics = Utils.compute_supervised_metrics(y_true=y_true, y_pred=y_pred)
        if p > 0:
            xp_supervised_metrics = {k:v*p for k,v in supervised_metrics.items()}
            xp_supervised_metrics = {f"{key}_allsamplesxp": value for key, value in xp_supervised_metrics.items()}

            best_solution_local = best_solution.loc[y_pred.index]
            assert best_solution_local.index.equals(y_pred.index)
            performance_metrics = Utils.compute_supervised_metrics(y_true=best_solution_local, y_pred=y_pred)
            performance_metrics = {f"{key}_performance": value for key, value in performance_metrics.items()}

            y_pred_artificial = pd.Series(np.arange(len(best_solution)), index= best_solution.index)
            y_pred_artificial += 5000
            y_pred_artificial.loc[y_pred.index] = y_pred

            assert y_true_total.index.equals(y_pred_artificial.index)
            artificial_supervised_metrics = Utils.compute_supervised_metrics(y_true=y_true_total, y_pred=y_pred_artificial)
            artificial_supervised_metrics = {f"{key}_artificial": value for key, value in artificial_supervised_metrics.items()}

            assert best_solution.index.equals(y_pred_artificial.index)
            artificial_performance_metrics = Utils.compute_supervised_metrics(y_true=best_solution, y_pred=y_pred_artificial)
            artificial_performance_metrics = {f"{key}_artificial_performance": value for key, value in artificial_performance_metrics.items()}

            supervised_metrics = {**supervised_metrics, **xp_supervised_metrics, **performance_metrics,
                                  **artificial_supervised_metrics, **artificial_performance_metrics}

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


