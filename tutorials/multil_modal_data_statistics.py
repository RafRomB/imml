"""
=============================================================
Statistics of a multi-modal dataset
=============================================================

A multi-modal dataset can be described using several measures.

In this tutorial, we will explore how to use `iMML` to describe a **multi-modal dataset**.

"""

# License: GNU GPLv3

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To make the figures you will need the following libraries installed: matplotlib.


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from sklearn.datasets import make_classification

from imml.utils import DatasetUtils
from imml.statistics import pid


##########################
# Step 2: Load the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^

random_state = 42
X, y = make_classification(random_state=random_state)
Xs = [X[:, :10], X[:, 10:]]
print("Samples:", len(Xs[0]), "\t", "Modalities:", len(Xs), "\t", "Features:", [X.shape[1] for X in Xs])


###################################################
# Step 3: Calculate PID statistics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using ``pid``, we can quantify the degree of redundancy, uniqueness, and synergy (PID statistics) relating input
# modalities with an output task.
rus = pid(Xs=Xs, y=y, random_state=random_state, normalize=True)
rus

###############################################################################
# You can visualize these metrics.
def overlap_area(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2)**2
    r1_2, r2_2 = r1*r1, r2*r2
    alpha = math.acos((d*d + r1_2 - r2_2) / (2*d*r1))
    beta  = math.acos((d*d + r2_2 - r1_2) / (2*d*r2))
    return r1_2*alpha + r2_2*beta - d*r1*math.sin(alpha)

def solve_distance_for_overlap(r1, r2, target_overlap, tol=1e-6, max_iter=100):
    lo = max(0.0, abs(r1 - r2))
    hi = r1 + r2
    max_overlap = math.pi * min(r1, r2)**2
    target_overlap = max(0.0, min(target_overlap, max_overlap))
    if target_overlap <= 0:
        return hi
    if abs(target_overlap - max_overlap) < tol:
        return lo
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        ov = overlap_area(r1, r2, mid)
        if abs(ov - target_overlap) < tol:
            return mid
        if ov > target_overlap:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def plot_two_circle_venn(counts, labels=("Set 1","Set 2"), colors=["white", "white", "white"], abb=True):
    a_only = round(float(counts.get("Uniqueness1", 0)), 2)
    b_only = round(float(counts.get("Uniqueness2", 0)), 2)
    inter  = round(float(counts.get("Redundancy", 0)), 2)
    outside = round(float(counts.get("Synergy", 0)), 2)
    A = a_only + inter
    B = b_only + inter
    r1 = math.sqrt(A / math.pi) if A>0 else 0.0
    r2 = math.sqrt(B / math.pi) if B>0 else 0.0
    d = solve_distance_for_overlap(r1, r2, inter)
    max_r = max(r1, r2)

    fig, ax = plt.subplots()

    side = math.sqrt(outside)
    rect = Rectangle((-r1/outside, -max_r/outside), (d+r2)/outside, 2*max_r/outside,
                     facecolor=colors[2], edgecolor="black", alpha=0.5)
    ax.add_patch(rect)

    ax.add_patch(Circle((0, 0), r1, facecolor=colors[0], alpha=0.5, edgecolor="black", linewidth=2))
    ax.add_patch(Circle((d, 0), r2, facecolor=colors[1], alpha=0.5, edgecolor="black", linewidth=2))

    if abb:
        u1, u2, r, s = "U", "U", "R", "S"
    else:
        u1, u2, r, s = "Uniqueness", "Uniqueness", "Redundancy", "Synergy"
    ax.text(-r1/2, 0, f"{u1}\n{a_only}", ha='center', va='center')
    ax.text(d + r2/2, 0, f"{u2}\n{b_only}", ha='center', va='center')
    ax.text(max_r -d/2, 0, f"{r}\n{inter}", ha='center', va='center')
    ax.text(max_r -d/2, max_r*1.2, f"{s} {outside}", ha='center', va='bottom')

    ax.text(0, -(max_r*1.1), labels[0], ha='center', va='top')
    ax.text(d, -(max_r*1.1), labels[1], ha='center', va='top')

    padding = max_r * 1.3 + d*0.1
    ax.set_xlim(-padding, d + padding)
    ax.set_ylim(-(max_r*1.6), max_r * 1.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return fig, ax

rus = {"Redundancy": 0.2, "Synergy": 0.1, "Uniqueness1": 0.45, "Uniqueness2": 0.25}
fig, ax = plot_two_circle_venn(rus, labels=("Modality A","Modality B"), colors=["#780000", "#669BBC", "#FDF0D5"], abb=False)


###################################################
# Step 4: Summary
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can also get a summary of the multi-modal dataset.
summary = DatasetUtils.get_summary(Xs=Xs)
pd.DataFrame(summary)

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we quantified redundancy, unique information, and synergy between modalities with respect to a
# target using PID, and we summarized key dataset statistics. These insights help understand complementarity between
# modalities and guide downstream multi-modal learning.
