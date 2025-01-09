import itertools
import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from reval.utils import kuhn_munkres_algorithm
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.multitest import multipletests
from tqdm.notebook import tqdm
from tqdm.contrib.itertools import product

from src.utils import GetMetrics


class ResultGenerator:


    @staticmethod
    def generate_results(results_path: str, supervised_metrics_path: str = None, inmetrics_path: str = None,
                         unsupervised_metrics_path: str = None, random_state = None,
                         n_permutations=1000, verbose=True, nb_workers=10, progress_bar=True):
        results = ResultGenerator.preprocess_results(results_path=results_path, nb_workers=nb_workers, verbose=verbose,
                                                     progress_bar=progress_bar)

        if unsupervised_metrics_path is not None:
            alg_uns_metrics = ResultGenerator.save_unsupervised_metrics(results=results,
                                                                        filepath=unsupervised_metrics_path,
                                                                        progress_bar=progress_bar)
        else:
            alg_uns_metrics = None
        if supervised_metrics_path is not None:
            supervised_metrics = ResultGenerator.save_supervised_metrics(results=results,
                                                                         filepath=supervised_metrics_path,
                                                                         random_state=random_state,
                                                                         n_permutations=n_permutations)
        else:
            supervised_metrics = None

        return results, alg_uns_metrics, supervised_metrics


    @staticmethod
    def save_alg_comparison(results: pd.DataFrame, filepath: str, progress_bar=True):
        comparison_columns = ["Dataset", "alg1", "alg2", "Run_N", "pred_alg1", "pred_alg2"]
        alg_comparisons = pd.DataFrame([], columns=comparison_columns)

        if progress_bar:
            iterator = product(results["Dataset"].unique(),
                    sorted(results["Run_N"].unique()),
                    set(itertools.combinations(results["Algorithm"].unique(), 2)))
        else:
            iterator = itertools.product(results["Dataset"].unique(),
                    sorted(results["Run_N"].unique()),
                    set(itertools.combinations(results["Algorithm"].unique(), 2)))

        for dataset, run_n, (alg1, alg2) in iterator:
            pred_alg1 = results.loc[(results["Dataset"] == dataset) &
                                    (results["Amputation_Mechanism"] == "Resampling") &
                                    (results["Run_N"] == run_n) &
                                    (results["Algorithm"] == alg1),
            ["Sorted_Y_Pred", "Y_Pred_Idx"]]
            pred_alg2 = results.loc[(results["Dataset"] == dataset) &
                                    (results["Amputation_Mechanism"] == "Resampling") &
                                    (results["Run_N"] == run_n) &
                                    (results["Algorithm"] == alg2),
            ["Sorted_Y_Pred", "Y_Pred_Idx"]]

            if pred_alg1.empty or pred_alg2.empty:
                continue
            pred_alg1 = pred_alg1.iloc[0]
            pred_alg2 = pred_alg2.iloc[0]
            mask = [idx for idx in pred_alg1["Y_Pred_Idx"] if idx in pred_alg2["Y_Pred_Idx"]]
            pred_alg1 = pd.Series(pred_alg1["Sorted_Y_Pred"], index=pred_alg1["Y_Pred_Idx"]).loc[mask]
            pred_alg2 = pd.Series(pred_alg2["Sorted_Y_Pred"], index=pred_alg2["Y_Pred_Idx"]).loc[mask]

            alg_comparison = pd.DataFrame([[dataset, alg1, alg2, run_n, pred_alg1, pred_alg2]],
                                          columns=comparison_columns)
            alg_comparisons = pd.concat([alg_comparisons, alg_comparison], ignore_index=True)

        alg_comparisons["AMI"] = alg_comparisons.apply(
            lambda x: adjusted_mutual_info_score(x["pred_alg1"], x["pred_alg2"]), axis=1)
        alg_comparisons["ARI"] = alg_comparisons.apply(
            lambda x: adjusted_rand_score(x["pred_alg1"], x["pred_alg2"]), axis=1)
        alg_comparisons["Overlapping"] = alg_comparisons.apply(
            lambda x: accuracy_score(x["pred_alg1"],
                                     kuhn_munkres_algorithm(true_lab=x["pred_alg1"], pred_lab=x["pred_alg2"])), axis=1)
        alg_comparisons = alg_comparisons.groupby(
            ["alg1", "alg2"], as_index=False)[["AMI", "ARI", "Overlapping"]].mean().sort_values("AMI", ascending=False)
        alg_comparisons["Comparison"] = alg_comparisons["alg1"] + "_" + alg_comparisons["alg2"]
        alg_comparisons = pd.concat([alg_comparisons,
                                     alg_comparisons.rename(columns={"alg1": "alg2", "alg2": "alg1"})],
                                    ignore_index=True)
        alg_comparisons.to_csv(filepath, index=None)

        return alg_comparisons


    @staticmethod
    def save_unsupervised_metrics(results: pd.DataFrame, filepath: str, random_state=None, progress_bar=True):
        alg_stability = results[['Dataset', "Algorithm", "Missing_Percentage", "Amputation_Mechanism", "Imputation",
                                 "Run_N", "Sorted_Y_Pred", "Y_Pred_Idx",
                                 'Silhouette', "VRC", "DB", "DBCV", "Dunn", "DHI", "SSEI", 'RSI', 'BHI']]

        alg_stability = alg_stability.loc[~(alg_stability["Amputation_Mechanism"] == "Resampling")]

        alg_uns_metrics = alg_stability.drop(columns=["Sorted_Y_Pred", "Run_N", "Y_Pred_Idx"])
        alg_uns_metrics = alg_uns_metrics.groupby(
            ["Dataset", "Algorithm", "Missing_Percentage", "Amputation_Mechanism", "Imputation"], as_index=False).mean()

        iterator = alg_stability["Dataset"].unique()
        if progress_bar:
            iterator = tqdm(iterator)

        for dataset in iterator:
            preds_dataset = alg_stability.loc[
                (alg_stability["Dataset"] == dataset), ["Missing_Percentage", "Algorithm", "Amputation_Mechanism",
                                                        "Imputation", "Run_N", "Sorted_Y_Pred", "Y_Pred_Idx"]]
            for alg in preds_dataset["Algorithm"].unique():
                pred_alg = preds_dataset[preds_dataset["Algorithm"] == alg]
                for missing_percentage in pred_alg["Missing_Percentage"].unique():
                    pred_missing_alg = pred_alg[pred_alg["Missing_Percentage"] == missing_percentage]
                    for amputation_mechanism in pred_missing_alg["Amputation_Mechanism"].unique():
                        pred_missing_ampt_alg = pred_missing_alg[
                            pred_missing_alg["Amputation_Mechanism"] == amputation_mechanism]
                        for impt in pred_missing_ampt_alg["Imputation"].unique():
                            pred_missing_ampt_impt_alg = pred_missing_ampt_alg[
                                pred_missing_ampt_alg["Imputation"] == impt]

                            amis, aris = [], []
                            for run_1, run_2 in set(itertools.combinations(pred_missing_ampt_impt_alg["Run_N"].unique(), 2)):
                                pred1_alg = pred_missing_ampt_impt_alg["Run_N"] == run_1
                                pred1_alg = pred_missing_ampt_impt_alg.loc[pred1_alg].iloc[0]
                                pred2_alg = pred_missing_ampt_impt_alg["Run_N"] == run_2
                                pred2_alg = pred_missing_ampt_impt_alg.loc[pred2_alg].iloc[0]
                                mask = [idx for idx in pred1_alg["Y_Pred_Idx"] if idx in pred2_alg["Y_Pred_Idx"]]
                                pred1_alg = pd.Series(pred1_alg["Sorted_Y_Pred"], index=pred1_alg["Y_Pred_Idx"]).loc[mask]
                                pred2_alg = pd.Series(pred2_alg["Sorted_Y_Pred"], index=pred2_alg["Y_Pred_Idx"]).loc[mask]
                                amis.append(adjusted_mutual_info_score(pred1_alg, pred2_alg)), aris.append(
                                    adjusted_rand_score(pred1_alg, pred2_alg))

                            alg_uns_metrics.loc[(alg_uns_metrics["Dataset"] == dataset) &
                                                (alg_uns_metrics["Missing_Percentage"] == missing_percentage) &
                                                (alg_uns_metrics["Amputation_Mechanism"] == amputation_mechanism) &
                                                (alg_uns_metrics["Imputation"] == impt) &
                                                (alg_uns_metrics["Algorithm"] == alg),
                            ["AMI_Sta", "ARI_Sta"]] = [np.mean(amis), np.mean(aris)]

        alg_uns_metrics.to_csv(filepath, index=None)
        return alg_uns_metrics


    @staticmethod
    def preprocess_results(results_path: str, verbose=True, nb_workers=10, progress_bar=True):
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

        mask = results["algorithm"] == "MONET"
        assert results[results["silhouette_excluding_outliers"].notnull() & (~mask)].empty
        results_missing = results[mask].copy()
        results_missing["y_pred"] = results_missing["y_pred"].str.replace("nan", "np.nan")
        results_missing["y_pred"] = results_missing["y_pred"].parallel_apply(lambda x: np.array(eval(x)))
        results_missing.loc[:, "algorithm"] = "MONET_IO"
        results_missing["y_pred"] = results_missing["y_pred"].apply(
            lambda x: pd.factorize(x, use_na_sentinel=False)[0].tolist())

        results_excluding_outliers = results[mask].copy()
        results_excluding_outliers["y_pred"] = results_excluding_outliers["y_pred"].str.replace("nan", "np.nan")
        results_excluding_outliers[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]] = results_excluding_outliers[
            ["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].applymap(lambda x: np.array(eval(x)))
        results_excluding_outliers.loc[:, "algorithm"] = "MONET_EO"
        metrics_col = ['silhouette', 'vrc', 'db', 'dbcv', 'dunn', "dhi", "ssei", 'rsi', 'bhi']
        results_excluding_outliers[metrics_col] = results_excluding_outliers[
            [f"{met}_excluding_outliers" for met in metrics_col]]
        results_excluding_outliers[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]] = results_excluding_outliers[
            ["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].apply(
            lambda row: (
                row["y_true"][~np.isnan(row["y_pred"])].astype(int).tolist(),
                row["y_pred"][~np.isnan(row["y_pred"])].astype(int).tolist(),
                row["y_true_idx"][~np.isnan(row["y_pred"])].astype(int).tolist(),
                row["y_pred_idx"][~np.isnan(row["y_pred"])].astype(int).tolist()),
            axis=1,
            result_type='expand')

        results_missing = pd.concat([results_missing, results_excluding_outliers])
        results_missing[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]] = results_missing[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].parallel_applymap(str)
        results = pd.concat([results.loc[~mask], results_missing]).reset_index(drop=True)
        results = results.drop(columns=[f"{met}_excluding_outliers" for met in metrics_col])

        results[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]] = results[
            ["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].parallel_applymap(eval)
        assert results["y_true_idx"].eq(results["y_pred_idx"]).all()

        results[["sorted_y_pred", "sorted_y_true"]] = results[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].parallel_apply(
            lambda x: (
                pd.Series(x["y_pred"], index=x["y_pred_idx"]).sort_index().to_list(),
                pd.Series(x["y_true"], index=x["y_true_idx"]).sort_index().to_list()),
            axis=1, result_type="expand")

        results["algorithm"] = results["algorithm"].str.upper()
        results.columns = results.columns.str.title()
        results = results.rename(columns={'Vrc': "VRC", 'Db': "DB", 'Dbcv': "DBCV", "Dhi": "DHI", "Ssei": "SSEI",
                                          "Rsi":'RSI', "Bhi": 'BHI'})
        
        

        return results


    @staticmethod
    def save_supervised_metrics(results: pd.DataFrame, filepath: str, random_state=None, n_permutations: int = 1000):
        supervised_metrics = results[["Sorted_Y_True", "Sorted_Y_Pred"]].parallel_apply(
            lambda row: GetMetrics.compute_supervised_metrics(y_true=row["Sorted_Y_True"], y_pred=row["Sorted_Y_Pred"],
                                                              random_state=random_state, n_permutations=n_permutations),
            axis=1)
        results = pd.concat([results, pd.DataFrame(supervised_metrics.to_dict()).T], axis=1)
        indexes_names = ["Dataset", "Algorithm", "Missing_Percentage", "Amputation_Mechanism", "Imputation"]
        results = results[results.select_dtypes(include="float").columns.to_list() + indexes_names].groupby(
            indexes_names, sort=False).agg(["mean", 'std', 'var']).reset_index()
        results.columns = results.columns.map('_'.join).str.strip('_')
        results.to_csv(filepath, index=None)
        return results

    @staticmethod
    def save_robustness_metrics(results: pd.DataFrame, filepath: str, random_state=None, n_permutations: int = 1000):
        labels_dict = {}
        for dataset in results["Dataset"].unique():
            mask = (results["Dataset"] == dataset) & (results["Amputation_Mechanism"] == "No")
            labels_dict[dataset] = {
                algorithm: results[(results["Algorithm"] == algorithm) & mask]["Sorted_Y_Pred"].to_list() for algorithm
                in results["Algorithm"].unique()}

        mask = results["Missing_Percentage"] != 0
        if results.loc[mask, "Imputation"].nunique() > 1:
            mask = mask & results["Imputation"]
        robustness_metrics = results.loc[mask].parallel_apply(
            lambda row: pd.DataFrame(
                [GetMetrics.compute_supervised_metrics(y_true=clusters_run_n, y_pred=row["Sorted_Y_Pred"],
                                                       random_state=random_state, n_permutations=n_permutations)
                 for clusters_run_n in labels_dict[row["Dataset"]][row["Algorithm"]]]).mean(), axis=1)
        results = pd.concat([results, robustness_metrics], axis=1)
        results.to_csv(filepath, index=None)
        return results

    # def save_stability_metrics(results: pd.DataFrame, filepath: str, random_state=None, progress_bar=True):
    #     results = pd.merge(results, pd.DataFrame(itertools.product(results["dataset"].unique(), results["Algorithm"].unique()),
    #                                     columns=["dataset", "Algorithm"]), how="right")
    #     res = OneHotEncoder(sparse_output=False).set_output(transform="pandas").fit_transform(
    #         results[["dataset", "Algorithm"]])
    #     for col in ["silhouette_mean", "silhouette_std", "MCC_mean", "MCC_std", "MCC (p-value)_mean",
    #                 "MCC (p-value)_std"]:
    #         res[col] = results[col]
    #         results[col] = KNNImputer().set_output(transform="pandas").fit_transform(X=res)[col]
    #         res = res.drop(columns=col)
    #     results["padj"] = multipletests(results["MCC (p-value)_mean"], method="fdr_bh")[1]
    #     results["log_padj"] = results["padj"].apply(lambda x: -np.log10(x))
    #     results.to_csv(inmetrics_path, index=None)
    #     outputs.append(results)
    #     return outputs



