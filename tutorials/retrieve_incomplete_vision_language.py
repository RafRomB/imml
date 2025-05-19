"""
=========================================================================
Retrieve an incomplete vision-language dataset.
=========================================================================

This tutorial demonstrates how to retrieve samples from an **incomplete vision-language dataset** using the `iMML`
library. `iMML` supports robust data retrieval even when some modalities (e.g., text or image) are missing, making it
suitable for real-world multi-modal data where missingness is common.

We will use the ``MCR`` algorithm from the `iMML` `retrieve` module on the Food101 dataset, simulating varying
degrees of missing modalities, and evaluate its performance under different scenarios.

"""

# sphinx_gallery_thumbnail_number = 1

# License: GNU GPLv3

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To run this tutorial, make sure the `deep` module of `iMML` is installed. To make the figures, you will also need
# the following libraries installed: matplotlib.


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from lightning import Trainer
import lightning as L
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats

from imml.classify import RAGPT
from imml.load import RAGPTDataset, RAGPTCollator
from imml.retrieve import MCR

################################
# Step 2: Prepare the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the Food101 dataset, a vision-language dataset for food recognition. This dataset can be
# downloaded from Kaggle at https://www.kaggle.com/datasets/gianmarco96/upmcfood101. For retrieval, we will
# use the ``MCR`` method from the `retrieve` module.

RANDOM_STATE = 42
L.seed_everything(RANDOM_STATE)

###############################################################################
# As we do not have the data in the repository, please, organize the folder as we did. First, create a folder with
# the name of the dataset, in this case, "food101". Next, put all the files there. You should have the following
# structure:
#  - A folder named `food101` containing:
#   - An `imgs` subfolder with images.
#   - A file named `train_titles.csv`.

main_folder = "food101"
folder_images = os.path.join(main_folder, "imgs2")
text_filename = "train_titles.csv"
text_path = os.path.join(main_folder, text_filename)

########################################
# We will convert the labels to numeric.
# While for the MCR we do not need training, we will still have a training set to facilitate its usage as a
# prompt generator for ``RAGPT``.
# Then, we will sample 10,000 image-text pairs and split into:
#  - 1,000 for training.
#  - 500 for validation.
#  - 5,000 for the memory bank.

train_text = pd.read_csv(text_path, header=None)
train_text.columns = ["img", "text", "label"]
train_text["img"] = train_text.apply(lambda x: os.path.join(folder_images, x["img"]), axis= 1)
le = LabelEncoder()
train_text["class"] = le.fit_transform(train_text["label"])
train_text = train_text.sample(10000, random_state=RANDOM_STATE)
# uncomment this line
# train_text["img"].apply(lambda x:
#                         Image.open(x).save(
#                             os.path.join(folder_images.replace("imgs", "imgs2"),
#                             os.path.basename(x))))
train_df = train_text.sample(1000, random_state=RANDOM_STATE)
test_df = train_text.drop(index=train_df.index).sample(500, random_state=RANDOM_STATE)
bank_df = train_text.drop(index=train_df.index). \
    drop(index=test_df.index).sample(5000, random_state=RANDOM_STATE)
print("train_df", train_df.shape)
train_df.head()

###############################################################################
train_df["label"].value_counts()


###################################################
# Step 3: Simulate missing modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To reflect realistic scenarios, we randomly introduce missing data. In this case, 70% of training and test samples
# will have either text or image missing. You can change this parameter for more or less amount of incompleteness.

p = 0.7
missing = np.random.default_rng(RANDOM_STATE).choice(train_df.index, size=int(len(train_df)*p/2))
train_df.loc[missing, "img"] = np.nan
missing = np.random.default_rng(RANDOM_STATE).choice(train_df.drop(index=missing).index, size=int(len(train_df)*p/2))
train_df.loc[missing, "text"] = np.nan

missing = np.random.default_rng(RANDOM_STATE).choice(test_df.index, size=int(len(test_df)*p/2))
test_df.loc[missing, "img"] = np.nan
missing = np.random.default_rng(RANDOM_STATE).choice(test_df.drop(index=missing).index, size=int(len(test_df)*p/2))
test_df.loc[missing, "text"] = np.nan


########################################################
# Step 4: Generate the memory bank
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

modalities = ["image", "text"]
batch_size = 64
n_neighbors = 10
estimator = MCR(batch_size=batch_size, modalities=modalities, generate_cap=True, save_memory_bank=True,
                prompt_path=main_folder, n_neighbors=n_neighbors)

Xs_bank = [
    bank_df["img"].to_list(),
    bank_df["text"].to_list()
]
y_bank = bank_df["class"]

# As this process can take a while, we will save the memory bank to a csv file.
# uncomment the lines
# estimator.fit(Xs=Xs_bank, y=y_bank)
# estimator.memory_bank.to_csv(os.path.join(main_folder, "memory_bank.csv"))
memory_bank = pd.read_csv(os.path.join(main_folder, "memory_bank.csv"), index_col=0)
for col in memory_bank.columns:
    try:
        memory_bank[col] = memory_bank[col].apply(eval)
    except: pass
print("memory_bank", memory_bank.shape)


########################################################
# Step 5: Retrieve
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xs_test = [
    test_df["img"].to_list(),
    test_df["text"].to_list()
]
y_test = test_df["class"]
# test_database = estimator.transform(Xs=Xs_test, y=y_test, memory_bank=memory_bank)
# test_database.to_csv(os.path.join(main_folder, f"test_database_missing{int(100*p)}.csv"))
test_database = pd.read_csv(os.path.join(main_folder, f"test_database_missing{int(100*p)}.csv"), index_col=0)
for col in test_database.columns:
    try:
        test_database[col] = test_database[col].apply(eval)
    except: pass
print("test_database", test_database.shape)


########################################################
# Step 5: Visualize the retrieved instances
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can visualize the top-2 retrieved instances for a given target sample.
# In fact, we can even approximate the target's label by selecting the most common label among the retrieved instances,
# similar to a k-nearest neighbors (KNN) approach.
#
# Let's begin by visualizing the top-2 retrieved instances for a target sample that is missing its text modality.
idx = 15
ex = test_database[
    test_database["observed_image"].astype(bool) &
    (~test_database["observed_text"].astype(bool))].iloc[idx]
images_to_show = []
image_to_show = ex["img_path"]
images_to_show.append(image_to_show)
print("Target:", image_to_show)
print("Label:", le.classes_[ex["label"]], "\t", "Text:", train_text.loc[ex.name, "text"])
retrieved_instances = memory_bank.loc[ex["i2i_id_list"][:2]]
for i,retrieved_instance in retrieved_instances.reset_index(drop=True).iterrows():
    image_to_show = retrieved_instance["img_path"]
    images_to_show.append(image_to_show)
    print(f"Top-{i+1} retrieved instance:", image_to_show)
    print("Label:", le.classes_[retrieved_instance["label"]], "\t",
          "Text:", retrieved_instance["text"])

print()
print("Real:", le.classes_[ex["label"]])
print("Prediction:", le.classes_[stats.mode(ex["i2i_label_list"])[0][0]])
images_to_show = [Image.open(image_to_show).resize((512, 512), Image.Resampling.LANCZOS)
                  for image_to_show in images_to_show]

fig, ax = plt.subplots(1,3, figsize=(6, 2), constrained_layout=True)
for i, image_to_show in enumerate(images_to_show):
    ax[i].axis("off")
    ax[i].imshow(image_to_show)

#################################
# Now, let’s consider a target instance that is missing its image modality.

idx = 4
ex = test_database[
    test_database["observed_text"].astype(bool) &
    (~test_database["observed_image"].astype(bool))].iloc[idx]
images_to_show = []
image_to_show = train_text.loc[ex.name, "img"]
images_to_show.append(image_to_show)
print("Target:", image_to_show)
print("Label:", le.classes_[ex["label"]], "\t", "Text:", ex["text"])
retrieved_instances = memory_bank.loc[ex["t2t_id_list"][:2]]
for i,retrieved_instance in retrieved_instances.reset_index(drop=True).iterrows():
    image_to_show = retrieved_instance["img_path"]
    images_to_show.append(image_to_show)
    print(f"Top-{i+1} retrieved instance:", image_to_show)
    print("Label:", le.classes_[retrieved_instance["label"]], "\t",
          "Text:", retrieved_instance["text"])

print()
print("Real:", le.classes_[ex["label"]])
print("Prediction:", le.classes_[stats.mode(ex["t2t_label_list"])[0][0]])
images_to_show = [Image.open(image_to_show).resize((512, 512), Image.Resampling.LANCZOS)
                  for image_to_show in images_to_show]

fig, ax = plt.subplots(1,3, figsize=(6, 2), constrained_layout=True)
for i, image_to_show in enumerate(images_to_show):
    ax[i].axis("off")
    ax[i].imshow(image_to_show)


###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We used the ``MCR`` retriever from the `iMML.retrieve` module to identify the most relevant instances from a
# memory bank. As shown in the examples, the retriever was able to retrieve highly similar instances to the target,
# even when one of the modalities (image or text) was missing.
#
# Moreover, the predicted label-based on the most common label among the top retrieved instances-matched the
# ground truth, demonstrating the retriever’s effectiveness.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This example illustrates how `iMML` enables robust retrieval and similarity search in vision-language datasets,
# even in the presence of missing modalities.