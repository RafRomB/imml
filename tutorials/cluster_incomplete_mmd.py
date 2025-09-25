"""
========================================
Clustering (in)complete multi-modal data
========================================

Clustering involves grouping samples into distinct groups. In this tutorial, we will explore how to use `iMML` to
perform clustering on an multi-modal dataset. `iMML` also supports clustering of incomplete multi-modal data,
allowing users to perform clustering without requiring complete data across all modalities.

"""

# sphinx_gallery_thumbnail_number = 1

# License: GNU GPLv3

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To make the figures you will need the tutorials module installed. For this, use this command in the
# terminal: pip install imml[tutorials].


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, adjusted_mutual_info_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from imml.preprocessing import MultiModTransformer, NormalizerNaN
from imml.ampute import Amputer
from imml.cluster import EEIMVC
from imml.visualize import plot_missing_modality


##########################
# Step 2: Load the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^
# For reproducibility, we generate a small synthetic classification dataset, and split the features into two
# modalities (Xs[0], Xs[1]). You can replace this section with your own data loading.

random_state = 42
X, y = make_classification(n_samples=100, random_state=random_state, n_clusters_per_class=1, n_classes=3)
X, y = pd.DataFrame(X), pd.Series(y)
# Two modalities: first 10 features and last 10 features
Xs = [X.iloc[:, :10], X.iloc[:, 10:]]
print("Samples:", len(Xs[0]), "\t", "Modalities:", len(Xs), "\t", "Features:", [X.shape[1] for X in Xs])
n_clusters = len(np.unique(y))
y.value_counts()


########################################################
# Step 3: Clustering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We show how to cluster the multi-modal data using `iMML`, in this case, using the algorithm ``EEIMVC``. For this
# example, we build a pipeline where we first normalize the data and then the samples are clustered.

pipeline = make_pipeline(
    MultiModTransformer(NormalizerNaN()),
    EEIMVC(n_clusters = n_clusters, random_state=random_state)
)
labels = pipeline.fit_predict(Xs)

###############################################################################
# Clustering performance is evaluated using the Adjusted Mutual Information (AMI) score, which measures the
# agreement between predicted clusters and the ground truth, independent of label permutations. We also plot a
# confusion matrix to visually assess the alignment between predicted clusters and true labels.
ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=labels)
print("Adjusted Mutual Information Score:", adjusted_mutual_info_score(labels_true=y, labels_pred=labels))
pd.Series(labels).value_counts()


###################################################
# Step 4: Simulate missing data (Amputation)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# As we mentioned, `iMML` can be used also for incomplete multi-modal learning, Using ``Amputer`` in `iMML`, we
# randomly introduce missing data to simulate a scenario where some modalities are missing. Here, 20% of
# the samples will be incomplete.

p = 0.2
amputed_Xs = Amputer(p= p, mechanism="mcar", random_state=42).fit_transform(Xs)

###############################################################################
# You can visualize which modalities are missing using a binary color map (white for missing modalities, while black
# indicates available modality).
plot_missing_modality(Xs=amputed_Xs)


########################################################
# Step 4: Clustering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, we repeat the clustering analysis, but this time, with the missing data.

pipeline = make_pipeline(
    MultiModTransformer(NormalizerNaN()),
    EEIMVC(n_clusters = n_clusters, random_state=random_state)
)
labels = pipeline.fit_predict(amputed_Xs)

ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=labels)
print("Adjusted Mutual Information Score:", adjusted_mutual_info_score(labels_true=y, labels_pred=labels))
pd.Series(labels).value_counts()


########################################################
# Step 5: Benchmarking
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now compare the performance of the clustering with and without missing data. We can also compare it with a
# simple baseline, where missing data is previously imputed with the average value per feature. We repeat the
# experiments 5 times across increasing missing data, to get more robust results.

ps = np.arange(0., 1., 0.2)
n_times = 10
methods = ["No previous imputing", "Baseline imputting"]
all_metrics = []

##############################################################################
for method in methods:
    for p in ps:
        missing_percentage = int(p*100)
        for i in range(n_times):
            pipeline = make_pipeline(
                MultiModTransformer(NormalizerNaN().set_output(transform="pandas")),
                EEIMVC(n_clusters=n_clusters, random_state=i))
            if method == "Baseline imputting":
                pipeline = make_pipeline(
                    MultiModTransformer(SimpleImputer().set_output(transform="pandas")),
                    *pipeline)
            pipeline = make_pipeline(Amputer(p=p, mechanism="mcar", random_state=i), *pipeline)
            clusters = pipeline.fit_predict(Xs)
            metric = adjusted_mutual_info_score(labels_true=y, labels_pred=clusters)
            result = {
                "Method": method,
                "Incomplete samples (%)": p,
                "Iteration": i,
                "AMI": metric,
            }
            all_metrics.append(result)

df = pd.DataFrame(all_metrics)
df = df.sort_values(["Method", "Incomplete samples (%)", "Iteration"], ascending=[False, True, True])
df.head()

###################################################################
g = df.groupby(["Method", "Incomplete samples (%)"])["AMI"]
stats = g.agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index()
mean_wide = stats.pivot(index="Incomplete samples (%)", columns="Method", values="mean")
sem_wide  = stats.pivot(index="Incomplete samples (%)", columns="Method", values="sem")
ax = mean_wide.plot(yerr=sem_wide, marker="o", capsize=3, ylabel="Adjusted mutual information")

###############################################################################
# The adjusted mutual information score indicates how well the clustering aligns with the ground truth. An AMI
# score closer to 1 indicates a strong match. The clustering solutions by ``IMSR`` and ``PIMVC``, as well as
# their respective baselines (where missing values were replaced with the feature-wise mean), are compared to
# ground truth topics across various missing rates (from 0% to 80%) for different data missing mechanisms.
# The mean of 50 repetitions with 95% confidence intervals are shown.

###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Both ``IMSR`` and ``PIMVC`` consistently outperformed their baselines across all missing rates and patterns.
# As expected, the clustering performance decreased when more incomplete samples were included. However, even with
# a high percentage of incomplete samples (>60%), ``IMSR`` and ``PIMVC`` produced meaningful clustering results in
# most cases, highlighting their robustness in handling incomplete multi-modal datasetes. Notably, ``IMSR``
# outperformed all other methods across every tested condition.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This case examples shows how `iMML` effectively supports clustering with incomplete multi-modal datasets,
# demonstrating the robustness and flexibility of `iMML` for clustering tasks in real-world multi-modal scenarios.

