"""
==========================================================================
Retrieval on a vision–language dataset (flickr8k)
==========================================================================

This tutorial demonstrates how to retrieve samples from an incomplete vision–language dataset using iMML.
We will use the MCR retriever to find similar items across modalities (image/text) even when one modality
is missing. The example uses the public nlphuji/flickr8k dataset from Hugging Face Datasets, so you don't
need to prepare files manually.

What you will learn:

- How to load a vision–language dataset.
- How to build a memory bank with MCR for cross-modal retrieval.
- How to retrieve relevant items with missing modalities.
- How to visualize top retrieved examples for qualitative inspection.

This tutorial is fully reproducible. You can swap the loading section with your own data by constructing two
parallel lists: image paths and texts for each sample.
"""
import shutil

# sphinx_gallery_thumbnail_number = 1

# License: GNU GPLv3

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To run this tutorial, install the extras for deep learning and tutorials:
#   pip install imml[deep]
# Additionally, we will use the Hugging Face Datasets library to load Flickr8k:
#   pip install datasets


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import lightning as L
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

from imml.retrieve import MCR

################################
# Step 2: Prepare the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the Flickr30k dataset, a public vision–language dataset with images and captions available on
# Hugging Face Datasets as nlphuji/flickr30k. For retrieval, we will use the MCR method from the retrieve module.

random_state = 42
L.seed_everything(random_state)
rng = np.random.default_rng(random_state)

# Local working directory (images will be saved here so MCR can read paths)
data_folder = "flickr30k"
folder_images = os.path.join(data_folder, "imgs")
os.makedirs(folder_images, exist_ok=True)

# Load the dataset
ds = load_dataset("nlphuji/flickr30k", split="test[:50]")

# Build a DataFrame with image paths and captions. We persist images to disk because the retriever expects paths.
n_total = len(ds)
rows = []
for i in range(n_total):
    ex = ds[i]
    img = ex.get("image", None)
    caption = ex.get("caption", ex.get("text", ""))[0]
    img_path = os.path.join(folder_images, f"{i:06d}.jpg")
    img.save(img_path)
    rows.append({"img": img_path, "text": caption})

df = pd.DataFrame(rows)

# Split into train and test sets
train_df = df.sample(int(n_total*0.8), random_state=random_state)
test_df = df.drop(index=train_df.index)
print("train_df", train_df.shape)
train_df.head()


###################################################
# Step 3: Simulate missing modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To reflect realistic scenarios, we randomly introduce missing data. In this case, 70% of test samples
# will have either text or image missing. You can change this parameter for more or less amount of incompleteness.

p = 0.7
missing_mask = test_df.sample(frac=p/2, random_state=random_state).index
test_df.loc[missing_mask, "img"] = np.nan
missing_mask = test_df.drop(labels=missing_mask).sample(n=len(missing_mask), random_state=random_state).index
test_df.loc[missing_mask, "text"] = np.nan


########################################################
# Step 4: Generate the memory bank
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

modalities = ["image", "text"]
batch_size = 64
estimator = MCR(batch_size=batch_size, modalities=modalities, save_memory_bank=True, generate_cap=True,
                prompt_path=data_folder)

Xs_train = [
    train_df["img"].to_list(),
    train_df["text"].to_list()
]
# Use dummy labels for API compatibility (labels are not provided in Flickr30k)
y_train = pd.Series(np.zeros(len(train_df)), index=train_df.index)

estimator.fit(Xs=Xs_train, y=y_train)


########################################################
# Step 5: Retrieve
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xs_test = [
    test_df["img"].to_list(),
    test_df["text"].to_list()
]
# Use dummy labels for API compatibility
y_test = pd.Series(np.zeros(len(test_df)), index=test_df.index)
test_db = estimator.transform(Xs=Xs_test, y=y_test, n_neighbors=2)
memory_bank = estimator.memory_bank_


########################################################
# Step 6: Visualize the retrieved instances
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can visualize the top-2 retrieved instances for a given target sample.
# Here we focus on qualitative inspection: looking at the images and reading the captions
# of the retrieved items to assess whether they are semantically similar to the target.
#
# Let's begin by visualizing the top-2 retrieved instances for a target sample that is missing its text modality.
idx = 0
ex = test_db[test_db["observed_image"].astype(bool) & (~test_db["observed_text"].astype(bool))].iloc[idx]
images_to_show = []
image_to_show = ex["img_path"]
images_to_show.append(image_to_show)
print("Target image:", image_to_show)
print("Target caption (missing):", df.loc[ex.name, "text"])  # original text for reference
retrieved_instances = memory_bank.loc[ex["i2i_id_list"]]
for i,retrieved_instance in retrieved_instances.reset_index(drop=True).iterrows():
    image_to_show = retrieved_instance["img_path"]
    images_to_show.append(image_to_show)
    print(f"Top-{i+1} retrieved instance:", image_to_show)
    print("Retrieved caption:", retrieved_instance["text"])

print()
images_to_show = [Image.open(image_to_show).resize((512, 512), Image.Resampling.LANCZOS)
                  for image_to_show in images_to_show]

fig, ax = plt.subplots(1,3, figsize=(6, 2), constrained_layout=True)
for i, image_to_show in enumerate(images_to_show):
    ax[i].axis("off")
    ax[i].imshow(image_to_show)

#################################
# Now, let’s consider a target instance that is missing its image modality.

idx = 0
ex = test_db[test_db["observed_text"].astype(bool) & (~test_db["observed_image"].astype(bool))].iloc[idx]
images_to_show = []
image_to_show = df.loc[ex.name, "img"]
images_to_show.append(image_to_show)
print("Target (missing image) caption:", ex["text"])
retrieved_instances = memory_bank.loc[ex["t2t_id_list"]]
for i,retrieved_instance in retrieved_instances.reset_index(drop=True).iterrows():
    image_to_show = retrieved_instance["img_path"]
    images_to_show.append(image_to_show)
    print(f"Top-{i+1} retrieved instance:", image_to_show)
    print("Retrieved caption:", retrieved_instance["text"])

print()
images_to_show = [Image.open(image_to_show).resize((512, 512), Image.Resampling.LANCZOS)
                  for image_to_show in images_to_show]

fig, ax = plt.subplots(1,3, figsize=(6, 2), constrained_layout=True)
for i, image_to_show in enumerate(images_to_show):
    ax[i].axis("off")
    ax[i].imshow(image_to_show)

shutil.rmtree(data_folder, ignore_errors=True)

###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We used the ``MCR`` retriever from the `iMML` to identify the most relevant instances from a
# memory bank. As shown in the examples, the retriever was able to retrieve highly similar instances to the target,
# even when one of the modalities (image or text) was missing.
#
# Beyond the images themselves, the retrieved captions provide additional context to judge semantic similarity.
# Even when one modality is missing, MCR retrieves items whose other modality (image or text) remains informative.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This example illustrates how `iMML` enables robust retrieval and similarity search in vision-language datasets,
# even in the presence of missing modalities.