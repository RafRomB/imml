"""
======================================================================================
Dimensionality reduction: Feature extraction and feature selection
======================================================================================

High-dimensional datasets can severely impact machine learning projects, often degrading performance due to the curse
of dimensionality, as well as the presence of correlated, noisy, or irrelevant features. These issues also increase
computational demands and reduce model interpretability. Consequently, reducing the number of features is often
critical. **Dimensionality reduction** addresses these challenges by enhancing computational efficiency, highlighting
key features, reducing noise and enabling better data visualization.

Dimensionality reduction encompasses two main approaches: feature selection and feature extraction.
**Feature selection** involves identifying the most relevant features from the dataset, while **feature extraction**
creates new features by transforming the original ones to capture essential information.

In this tutorial, we will explore how to use ``JNMF`` to perform feature selection and feature extraction on an
**incomplete multi-modal dataset**. You will learn to handle missing data, infer modality importance, and visualize
the contributions of top features.

"""

# License: GNU GPLv3

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To run this tutorial and generate the figures, install the extra dependencies:
#   pip install imml[tutorials]


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import os

from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches

from imml.decomposition import JNMF
from imml.preprocessing import MultiModTransformer, ConcatenateMods
from imml.ampute import Amputer
from imml.feature_selection import JNMFFeatureSelector


###################################
# Step 2: Define plotting functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def get_modality_importance(Xs, selected_features, weights, names):
    selected_features = {"Feature": selected_features, "Feature Importance": weights}
    selected_features = pd.DataFrame(selected_features)
    selected_features = selected_features.sort_values(by="Feature Importance",
                                                      ascending=False)
    selected_features["Modality"] = selected_features["Feature"].apply(
        lambda x: [name for X,name in zip(Xs, names) if x in X.columns][0])
    selected_features = selected_features.groupby("Modality")["Feature Importance"].sum()
    selected_features = selected_features.div(selected_features.sum()).mul(100)
    selected_features = selected_features.sort_values(ascending=False)
    return selected_features

def get_top_features(Xs, selected_features, weights, components, names):
    selected_features = {"Feature": selected_features, "Feature Importance": weights,
                         "Component": components}
    selected_features = pd.DataFrame(selected_features)
    selected_features = selected_features.sort_values(by="Feature Importance",
                                                      ascending=False)
    selected_features["Modality"] = selected_features["Feature"].apply(
        lambda x: [name for X,name in zip(Xs, names) if x in X.columns][0])
    selected_features["Component"] += 1
    return selected_features

def get_contributions(Xs, selected_features, weights, components, names):
    selected_features = {"Feature": selected_features, "Feature Importance": weights,
                         "Component": components}
    selected_features = pd.DataFrame(selected_features)
    selected_features = selected_features.sort_values(by=["Component", "Feature Importance"],
                                                      ascending=[True, False])
    selected_features["Modality"] = selected_features["Feature"].apply(
        lambda x: [name for X,name in zip(Xs, names) if x in X.columns][0])
    sep = list(range(selected_features["Component"].max())) * pipeline[-1].n_components
    selected_features["Hue"] = sep
    selected_features["Component"] += 1
    return selected_features


##########################
# Step 3: Load the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^
# For reproducibility, we generate a small synthetic classification dataset and split the features into two
# modalities (Xs[0], Xs[1]).
# Optional: set a random_state for reproducibility (we do below).
#
# Using your own data:
#
# - Represent your dataset as a Python list Xs, one entry per modality.
# - Each Xs[i] should be a 2D array-like (pandas DataFrame or NumPy array) of shape (n_samples, n_features_i).
# - All modalities must refer to the same samples and be aligned by row order or index.

random_state = 42
X, y = make_classification(n_samples=50, random_state=random_state, n_clusters_per_class=1, n_classes=3)
X, y = pd.DataFrame(X), pd.Series(y)
X.columns = X.columns.astype(str)
# Two modalities: first 10 features and last 10 features
Xs = [X.iloc[:, :10], X.iloc[:, 10:]]
print("Samples:", len(Xs[0]), "\t", "Modalities:", len(Xs), "\t", "Features:", [X.shape[1] for X in Xs])
n_clusters = len(np.unique(y))
y.value_counts()

########################################################
# Step 4: Apply feature selection and feature extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_components = 4
# Feature extraction
pipeline = make_pipeline(MultiModTransformer(MinMaxScaler().set_output(transform="pandas")),
                         JNMF(n_components=n_components, random_state=random_state))
pipeline.fit(Xs)

# Feature selection.
pipeline = make_pipeline(MultiModTransformer(MinMaxScaler().set_output(transform="pandas")),
                         JNMFFeatureSelector(n_components=n_components, random_state=random_state))
pipeline.fit(Xs)

########################################################
# We can identify and visualize the selected features.
selected_features = get_top_features(Xs=Xs, selected_features= pipeline[-1].selected_features_,
                                     weights= pipeline[-1].weights_, components= pipeline[-1].component_,
                                     names= ["Modality A", "Modality B"])
selected_features

palette = {mod:col for mod, col in zip(selected_features.index, ["#2ca25f", "#99d8c9"])}
palette_list = [palette[mod] for mod in selected_features["Modality"]]
plt.figure(figsize= (4, 3))
ax = sns.barplot(data=selected_features, y="Component", x="Feature Importance",
                 legend=False, orient="h", order= selected_features["Component"],
                 )
ax.set_xlim(0, selected_features["Feature Importance"].max() + .8)

col = 0
for x in ax.properties()['children']:
    if isinstance(x, matplotlib.patches.Rectangle):
        x.set_color(palette_list[col])
        col += 1
    if col == len(selected_features):
        break

for i, container in enumerate(ax.containers):
    ax.bar_label(container, labels=selected_features["Feature"], padding = 3)

###############################################################################
# The top features selected attributes from both gene expression and fatty acid measurements, with two genes and
# two fatty acid identified as the most significant, forming a unified feature panel.

###################################################################
# We can visualize the modality relative importance with a barplot.
selected_features = get_modality_importance(
    Xs=Xs, selected_features= pipeline[-1].selected_features_,
    weights= pipeline[-1].weights_, names= ["Modality A", "Modality B"])
selected_features.to_frame()

ax = selected_features.plot(kind= "bar", color= list(palette.values()),
                            figsize= (3, 3), ylabel= "Modality Importance (\%)",
                            rot=0)
handles = [mpatches.Patch(color=color, label=modality)
           for modality, color in palette.items()]
_ = ax.legend(handles=handles, title="Modality", loc='best')

###############################################################################
# The genes seem to be the most important modality.

# ###############################################################################
# # We can also extract features and visualize the original features with the largest contribution to the components.
#
# pipeline = make_pipeline(MultiModTransformer(MinMaxScaler().set_output(transform="pandas")),
#                          JNMFFeatureSelector(n_components = n_components, select_by="component",
#                                              random_state=42, f_per_component=3))
# pipeline.fit(amputed_Xs)
# selected_features = get_contributions(Xs=Xs, selected_features= pipeline[-1].selected_features_,
#                                      weights= pipeline[-1].weights_, components= pipeline[-1].component_,
#                                      names= names)
# selected_features.drop(columns= "Hue")
#
# ###################################################################
#
# plt.figure(figsize= (4, 3))
# ax = sns.barplot(data=selected_features, y="Component", x="Feature Importance",
#                  hue="Hue", legend=False, orient="h", width= .9,
#                  )
#
# ax.set_xlim(0, selected_features["Feature Importance"].max() + .8)
#
# selected_features = selected_features.sort_values("Hue")
# col = 0
# for i, x in enumerate(ax.properties()['children']):
#     if isinstance(x, matplotlib.patches.Rectangle):
#         x.set_color(palette[selected_features["Modality"].iloc[col]])
#         col += 1
#     if col == len(selected_features):
#         break
#
# for i, container in enumerate(ax.containers):
#     ax.bar_label(container, labels=selected_features[selected_features["Hue"] == i]["Feature"],
#                  padding = 3)
#
#
# ########################################################
# # Step 6: Benchmarking
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # We simulate various block- and feature-wise missing data patterns. For feature selection, the most influential
# # feature from each component was chosen. To provide comparative benchmarks, we included baselines using randomly
# # selected features and all available features. The outputs from these methods were then used as inputs for a
# # support vector machine to predict genetic type. As the feature selection process does not replace missing values,
# # an imputation step was applied prior the classification. We also evaluate the effect of the number of
# # components/features on performance. We repeat the analysis 50 times with different seeds to have robust results.
# #
#
# ps = np.arange(0, 0.9, 0.2)
# n_components_list = [1, 2, 4, 8, 16]
# n_times = 50
# algorithms = ["Feature extraction", "Feature selection", "Randomly selected features", "All features"]
# mechanisms = ["mem", "pm", "mcar", "mnar"]
# all_metrics = {}
#
# ########################################################
# # For making the comparison, uncomment the following code.
#
# # for algorithm in tqdm(algorithms):
# #     all_metrics[algorithm] = {}
# #     for mechanism in tqdm(mechanisms):
# #         all_metrics[algorithm][mechanism] = {}
# #         for n_components in n_components_list:
# #             all_metrics[algorithm][mechanism][n_components] = {}
# #             for p in ps:
# #                 missing_percentage = int(p*100)
# #                 all_metrics[algorithm][mechanism][n_components][missing_percentage] = {}
# #                 for i in range(n_times):
# #                     all_metrics[algorithm][mechanism][n_components][missing_percentage][i] = {}
# #                     Xs_train = Amputer(p= p, random_state=i).fit_transform(Xs)
# #                     for X in Xs_train:
# #                         X.iloc[np.random.default_rng(i).choice([True, False], p= [p,1-p], size = X.shape)] = np.nan
# #                     if algorithm == "Feature extraction":
# #                         pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")),
# #                                                  JNMF(n_components = n_components, random_state=i),
# #                                                  )
# #                     elif algorithm == "Feature selection":
# #                         pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")),
# #                                                  JNMFFeatureSelector(n_components = n_components, random_state=i),
# #                                                  FunctionTransformer(lambda x: np.concatenate(x, axis=1)),
# #                                                  SimpleImputer(),
# #                                                  )
# #                     elif algorithm == "Randomly selected features":
# #                         pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")),
# #                                                  ConcatenateViews(),
# #                                                  SimpleImputer().set_output(transform="pandas"),
# #                                                  FunctionTransformer(lambda x:
# #                                                                      x.iloc[:,np.random.default_rng(i).integers(0,
# #                                                                                                                 sum([X.shape[1] for X in Xs_train]),
# #                                                                                                                 size= n_components)]),
# #                          )
# #                     elif algorithm == "All features":
# #                         pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")),
# #                                                  ConcatenateViews(),
# #                                                  SimpleImputer().set_output(transform="pandas"),
# #                          )
# #                     try:
# #                         transformed_X = pipeline.fit_transform(Xs_train)
# #                         preds = SVC(random_state=i).fit(transformed_X, y).predict(transformed_X)
# #                         metric = accuracy_score(y_pred=preds, y_true=y)
# #                         all_metrics[algorithm][mechanism][n_components][missing_percentage][i]["Accuracy"] = metric
# #                         all_metrics[algorithm][mechanism][n_components][missing_percentage][i]["Comments"] = ""
# #                     except Exception as ex:
# #                         all_metrics[algorithm][mechanism][n_components][missing_percentage][i]["Comments"] = ex
# #
# # flattened_data = [
# #     {
# #         'Method': algorithm,
# #         'Mechanism': mechanism,
# #         'Missing rate (\%)': p,
# #         'Components': n_components,
# #         'Iteration': i,
# #         **iter_dict
# #     }
# #     for algorithm, algorithm_dict in all_metrics.items()
# #     for mechanism, mechanism_dict in algorithm_dict.items()
# #     for n_components, n_components_dict in mechanism_dict.items()
# #     for p, p_dict in n_components_dict.items()
# #     for i, iter_dict in p_dict.items()
# # ]
# # df = pd.DataFrame(flattened_data)
# # df['Method'] = pd.Categorical(df['Method'], categories=["Feature extraction", "Feature selection", "All features", "Randomly selected features"], ordered=True)
# # df = df.sort_values(["Method", "Mechanism", "Missing rate (\%)", "Components", "Iteration"], ascending=[True, False, True, True, True])
# # df.to_csv("reduction_results.csv", index= None)
# # print(df.shape)
# # df.head()
#
# ###################################################################
#
# df = pd.read_csv("reduction_results.csv")
# print(df.shape)
# df.head()
#
# ###################################################################
# # Let's see if there was any error during the benchmarking.
#
# errors = df[df["Comments"].notnull()]
# print("errors", errors.shape)
# errors
#
# ################################################################################
# # 0.1% models got errors. They were due to the fact that, after simulating high missing rates and randomly
# # select features, none of them has any available value, so the model could not be computed.
# #
# # Let's now visualize the results.
#
# mechanism_names = {"mem": "Mutually exclusive missing", "pm": "Partial missing",
#                    "mnar": "Missing not at random", "mcar": "Missing completely at random"}
# colorblind_palette = sns.color_palette("colorblind")
# g = sns.FacetGrid(data=df, col="Mechanism", row="Components", despine=False)
# g = g.map_dataframe(sns.pointplot, x="Missing rate (\%)", y="Accuracy", hue="Method",
#                     linestyles=["-", "--", ":", "-."], capsize= 0.05,
#                     seed= 42, palette=colorblind_palette)
# handles = [plt.Line2D([0], [0], color=col, lw=2, linestyle=linestyle)
#                   for col,linestyle in zip(colorblind_palette, ["-", "--", ":", "-."])]
# g.axes[0][0].legend(handles=handles, labels=df["Method"].unique().tolist(), loc= "best")
#
# for axes,n_components in zip(g.axes, df["Components"].unique()):
#     for ax,mechanism in zip(axes, df["Mechanism"].unique()):
#         ax.set_title(f"{mechanism.upper()}, Components\(|\)Features = {n_components}")
#
# plt.tight_layout()
#
# ###############################################################################
#
# ###################################
# # Summary of results
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # Accuracy was particularly low when using only a few components in the extracted features, likely because ``JNMF``, as
# # an unsupervised method, may capture patterns unrelated to the target label. In general, across all missing
# # rates, ``JNMF`` -both for feature extraction and selection- outperformed the baseline.
# #
# # For lower to medium number of component/ features, the selected features achieved impressive accuracy, particularly
# # when the missing data were not high. As the number of components increased, the extracted features consistently
# # outperformed other reducing methods. Thus, extracted features demonstrated robustness, especially in scenarios with
# # higher levels of missing data, where they surpassed selected features. This was largely because the selected
# # features required imputation post-selection before classification, introducing noise and potential inaccuracies.
#
# ###################################
# # Conclusion
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # This case demonstrates the power of `iMML` for feature selection and extraction, key steps in multiple domains,
# # such as biomedicine, where identifying biomarkers is crucial for personalized medicine. The reduced feature
# # matrices can also be used for other downstream tasks, such as classification or regression, as also shown here,
# # supporting further potential of the package for multi-modal learning.
#
