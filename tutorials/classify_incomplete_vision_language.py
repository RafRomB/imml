"""
=========================================================================
Classify an incomplete vision–language dataset (Oxford‑IIIT Pets) with deep learning
=========================================================================

This tutorial demonstrates how to classify samples from an incomplete vision–language dataset using the `iMML`
library. `iMML` supports robust classification even when some modalities (e.g., text or image) are missing, making it
suitable for real‑world multi‑modal data where missingness is common.

We will use the ``RAGPT`` algorithm from the `iMML` classify module on the Oxford‑IIIT Pets dataset and evaluate performance.

What you will learn:

- How to load a public vision–language dataset (Oxford‑IIIT Pets via Hugging Face Datasets).
- How to adapt this workflow to your own vision–language data.
- How to build a retrieval‑augmented memory bank and prompts with ``MCR``.
- How to train the ``RAGPT`` classifier when image or text may be missing.
- How to track metrics during training and evaluate with MCC and a confusion matrix.
"""

# sphinx_gallery_thumbnail_number = 1

# License: BSD 3-Clause License

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To run this tutorial, install the extras for deep learning and tutorials:
#   pip install imml[deep]
# We also use the Hugging Face Datasets library to load Oxford‑IIIT Pets:
#   pip install datasets


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import shutil
from lightning import Trainer
import lightning as L
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

from imml.classify import RAGPT
from imml.load import RAGPTDataset, RAGPTCollator
from imml.retrieve import MCR

################################
# Step 2: Prepare the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the oxford-iiit-pet-vl-enriched dataset, a public vision–language dataset with images and captions
# available on Hugging Face Datasets as visual-layer/oxford-iiit-pet-vl-enriched. For retrieval, we will use
# the ``MCR`` class from the retrieve module.

random_state = 42
L.seed_everything(random_state)
rng = np.random.default_rng(random_state)

# Local working directory (images will be saved here so ``MCR`` can read paths)
data_folder = "oxford_iiit_pet"
folder_images = os.path.join(data_folder, "imgs")
os.makedirs(folder_images, exist_ok=True)

# Load the dataset
ds = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched", split="train[:50]")

# Build a DataFrame with image paths and captions. We persist images to disk because
# the retriever expects paths.
n_total = len(ds)
rows = []
for i in range(n_total):
    ex = ds[i]
    img = ex.get("image", None)
    caption = ex.get("caption_enriched", None)[0]
    label = ex.get("label_cat_dog", None)[0]
    img_path = os.path.join(folder_images, f"{i:06d}.jpg")
    try:
        img.save(img_path)
    except Exception:
        img.convert("RGB").save(img_path)
    rows.append({"img": img_path, "text": caption, "label": label})

df = pd.DataFrame(rows)
le = LabelEncoder()
df["class"] = le.fit_transform(df["label"])
df["class"].value_counts()

###################################
# Split into 40% bank memory, 40% train and 20% test sets
bank_df = df.sample(int(n_total*0.4), random_state=random_state)
train_df = df.drop(index=bank_df.index).sample(n=len(bank_df), random_state=random_state)
test_df = df.drop(index=bank_df.index).drop(index=train_df.index)
print("train_df", train_df.shape)
train_df.head()


###################################################
# Step 3: Simulate missing modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To reflect realistic scenarios, we randomly introduce missing data. In this case, 30% of training and test samples
# will have either text or image missing. You can change this parameter for more or less amount of incompleteness.

p = 0.3
missing_mask = train_df.sample(frac=p/2, random_state=random_state).index
train_df.loc[missing_mask, "img"] = np.nan
missing_mask = train_df. \
    drop(labels=missing_mask). \
    sample(n=len(missing_mask), random_state=random_state). \
    index
train_df.loc[missing_mask, "text"] = np.nan

missing_mask = test_df.sample(frac=p/2, random_state=random_state).index
test_df.loc[missing_mask, "img"] = np.nan
missing_mask = test_df. \
    drop(labels=missing_mask). \
    sample(n=len(missing_mask), random_state=random_state). \
    index
test_df.loc[missing_mask, "text"] = np.nan


########################################################
# Step 4: Generate the prompts using a retriever
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the ``MCR`` (Multi-Channel Retriever) to construct a memory bank and generate prompts for the ``RAGPT`` model.

modalities = ["image", "text"]
batch_size = 64
estimator = MCR(batch_size=batch_size, modalities=modalities, save_memory_bank=True,
                prompt_path=data_folder, n_neighbors=5, generate_cap=True)

Xs_bank = [
    bank_df["img"].to_list(),
    bank_df["text"].to_list()
]
y_bank = bank_df["class"]

estimator.fit(Xs=Xs_bank, y=y_bank)
print("memory_bank", estimator.memory_bank_.shape)
estimator.memory_bank_.head()

########################################################
# Load generated training and testing prompts.

Xs_train = [
    train_df["img"].to_list(),
    train_df["text"].to_list()
]
y_train = train_df["class"]
train_db = estimator.transform(Xs=Xs_train, y=y_train)
print("train_db", train_db.shape)
train_db.head()

Xs_test = [
    test_df["img"].to_list(),
    test_df["text"].to_list()
]
y_test = test_df["class"]
test_db = estimator.transform(Xs=Xs_test, y=y_test)
print("test_db", test_db.shape)


########################################################
# Step 5: Training the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create the loaders.
train_data = RAGPTDataset(database=train_db)
train_dataloader = DataLoader(dataset= train_data, batch_size=batch_size,
                              collate_fn= RAGPTCollator(), shuffle=True)

test_data = RAGPTDataset(database=test_db)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                             collate_fn=RAGPTCollator(), shuffle=False)

########################################################
# Train the ``RAGPT`` model using the generated prompts. For speed in this demo we train for 2 epochs using
# Lightning.
trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
estimator = RAGPT(cls_num=len(le.classes_))
trainer.fit(estimator, train_dataloader)

########################################################
# Step 6: Advanced Usage: Track Metrics During Training
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can modify the internal functions. For instance, we can track loss and compute evaluation metrics during
# training.

trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
estimator = RAGPT(cls_num=len(le.classes_))
estimator.loss_list = []
estimator.agg_loss_list = []
validation_step = estimator.validation_step

def compute_metric(*args):
    loss = validation_step(*args)
    estimator.loss_list.append(loss)
    return loss
estimator.validation_step = compute_metric

def agg_metric(*args):
    estimator.agg_loss_list.append(torch.stack(estimator.loss_list).mean())
    estimator.loss_list = []
estimator.on_validation_epoch_end = agg_metric

trainer.fit(estimator, train_dataloader, test_dataloader)

#######################################################
# After training, we can evaluate predictions.
preds = trainer.predict(estimator, test_dataloader)
preds = [batch.softmax(dim=1) for batch in preds]
preds = [pred for batch in preds for pred in batch]
preds = torch.stack(preds).argmax(1).cpu()
losses = [i.item() for i in estimator.agg_loss_list]
shutil.rmtree(data_folder, ignore_errors=True)

#######################################################

ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=preds)
print("Testing metric:", matthews_corrcoef(y_true=y_test, y_pred=preds))

###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We first built a memory bank with 20 independent vision-language samples using the `iMML` ``retrieve`` module to
# generate retrieval-augmented prompts with a multi-channel retriever (``MCR``). Subsequently, we trained a model
# using the ``RAGPT`` algorithm available in `iMML` under 30% randomly missing text and image modalities. The model
# demonstrated strong robustness on the test set.
#
# This example is intentionally simplified, using only 50 instances for demonstration.
# For stronger performance and more reliable results, the full dataset and longer training should be used.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This example illustrates how `iMML` enables state-of-the-art performance in classification, even in the presence
# of significant modality incompleteness in vision-language datasets.

