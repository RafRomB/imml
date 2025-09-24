"""
=============================================================
Statistics and interaction structure of a multi-modal dataset
=============================================================

A multi-modal dataset can be characterized beyond basic shape information. With `iMML` you can:

- *Summarize core properties* of each modality (samples, features, completeness).
- *Quantify* how modalities relate to a target via PID (Partial Information Decomposition):
  Redundancy (shared info), Uniqueness (modality-specific info), and Synergy (info emerging only when modalities are combined).

In this tutorial, we will explore how to use `iMML` to describe a multi-modal dataset, compute PID statistics, and
visualize them effectively. We will also discuss tips to interpret the results.

"""

# sphinx_gallery_thumbnail_number = 2

# License: GNU GPLv3

###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import copy
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from imml.statistics import pid, get_summary
from imml.visualize import plot_pid

###################################
# Step 2: Create or load a multi-modal dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For reproducibility, we generate a small synthetic classification dataset, and split the features into two
# modalities (Xs[0], Xs[1]). You can replace this section with your own data loading.

random_state = 42
X, y = make_classification(n_samples=50, random_state=random_state)
# Two modalities: first 10 features and last 10 features
Xs = [X[:, :10], X[:, 10:]]
print("Samples:", len(Xs[0]), "\t", "Modalities:", len(Xs), "\t", "Features:", [X.shape[1] for X in Xs])


###################################################
# Step 3: Summarize the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The get_summary function provides a compact overview of the multi-modal dataset. We can make the dataset a bit
# more complex by introducing some incomplete samples.

inc_Xs = copy.deepcopy(Xs)
inc_Xs[0][:20, :] = np.nan
inc_Xs[0][25, 1] = np.nan
inc_Xs[1][18:22, :] = np.nan
inc_Xs[1][-15:, 3] = np.nan
summary = get_summary(Xs=inc_Xs, modalities=["Modality A", "Modality B"])
summary = pd.DataFrame(summary).astype(int)
summary.T

###################################################

summary.index = summary.index.str.replace(" samples", "")
_ = summary.plot(kind="bar", xlabel="Samples", ylabel="Count", rot=0,
                 title="Summary of the multi-modal dataset")


###################################################
# Step 4: Compute PID statistics (Redundancy, Uniqueness, Synergy)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using pid, we quantify the degree of redundancy, uniqueness, and synergy relating input modalities to the target.

rus = pid(Xs=Xs, y=y, random_state=random_state, normalize=True)
rus  # a dict with keys: Redundancy, Uniqueness1, Uniqueness2, Synergy


###############################################################################
# Step 5: Visualize the PID as a Venn-like diagram
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can directly pass the rus dict returned by pid to plot_pid. Alternatively, plot_pid can also compute pid
# internally if you pass Xs and y, which is convenient when you want a one-liner.

rus = {"Redundancy": 0.2, "Synergy": 0.1, "Uniqueness1": 0.45, "Uniqueness2": 0.25}
fig, ax = plot_pid(rus=rus, modalities=["Modality A", "Modality B"], abb=False)

###################################################
# Step 6: Interpreting PID results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# - Redundancy: Information about the target available in both modalities. High values suggest overlap.
# - Uniqueness1/2: Modality-specific information about the target. High values suggest complementarity.
# - Synergy: Information that emerges only when modalities are combined. High synergy often indicates interactions.
# If redundancy is high while uniqueness and synergy are low, this may suggest that the dataset could be more
# appropriately analyzed using classical unimodal modeling.

###################################################
# Step 7: Working with more than two modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If you have more than two modalities, PID statistics are computed pairwise and the pid function returns a list of
# dicts (one per pair).
rus = pid(Xs=Xs + [Xs[0]], y=y, random_state=random_state, normalize=True)
rus


###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we:
#
# - Summarized key per-modality statistics for a multi-modal dataset.
# - Quantified redundancy, uniqueness, and synergy with respect to a target using PID.
# - Visualized and interpreted PID, including the multi-modality (>2) case.
#
# These insights help you understand complementarity and interactions across modalities, informing model design and
# feature engineering for downstream multi-modal learning.
