"""
=========================================================================
Classify an incomplete vision-language dataset using deep learning.
=========================================================================

This tutorial demonstrates how to classify samples from an **incomplete vision-language dataset** using the `iMML`
library. `iMML` supports robust classification even when some modalities (e.g., text or image) are missing, making it
suitable for real-world multi-modal data where missingness is common.

We will use the ``RAGPT`` algorithm from the `iMML` `classify` module on the Food101 dataset, simulating varying
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
# downloaded from Kaggle at https://www.kaggle.com/datasets/gianmarco96/upmcfood101. For classification, we will
# use the ``RAGPT`` algorithm from the `classify` module.

torch.set_float32_matmul_precision('medium')
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
# To reflect realistic scenarios, we randomly introduce missing data. In this case, 30% of training and test samples
# will have either text or image missing. You can change this parameter for more or less amount of incompleteness.

p = 0.3
missing = np.random.default_rng(RANDOM_STATE).choice(train_df.index, size=int(len(train_df)*p/2))
train_df.loc[missing, "img"] = np.nan
missing = np.random.default_rng(RANDOM_STATE).choice(train_df.drop(index=missing).index, size=int(len(train_df)*p/2))
train_df.loc[missing, "text"] = np.nan

missing = np.random.default_rng(RANDOM_STATE).choice(test_df.index, size=int(len(test_df)*p/2))
test_df.loc[missing, "img"] = np.nan
missing = np.random.default_rng(RANDOM_STATE).choice(test_df.drop(index=missing).index, size=int(len(test_df)*p/2))
test_df.loc[missing, "text"] = np.nan


########################################################
# Step 4: Generate the prompts using a retriever
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the ``MCR`` (Multi-Channel Retriever) to construct a memory bank and generate prompts for the ``RAGPT`` model.

modalities = ["image", "text"]
batch_size = 64
n_neighbors = 10
estimator = MCR(batch_size=batch_size, modalities=modalities, save_memory_bank=True,
                prompt_path=main_folder, n_neighbors=n_neighbors)

Xs_bank = [
    bank_df["img"].to_list(),
    bank_df["text"].to_list()
]
y_bank = bank_df["class"]

########################################################
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
# Load generated training and testing prompts.

Xs_train = [
    train_df["img"].to_list(),
    train_df["text"].to_list()
]
y_train = train_df["class"]
# uncomment the lines
# train_database = estimator.transform(Xs=Xs_train, y=y_train, memory_bank=memory_bank)
# train_database.to_csv(os.path.join(main_folder, f"train_database_missing{int(100*p)}.csv"))
train_database = pd.read_csv(os.path.join(main_folder, f"train_database_missing{int(100*p)}.csv"), index_col=0)
for col in train_database.columns:
    try:
        train_database[col] = train_database[col].apply(eval)
    except: pass
print("train_database", train_database.shape)
train_database.head()

Xs_test = [
    test_df["img"].to_list(),
    test_df["text"].to_list()
]
y_test = test_df["class"]
# uncomment the lines
# test_database = estimator.transform(Xs=Xs_test, y=y_test, memory_bank=memory_bank)
# test_database.to_csv(os.path.join(main_folder, f"test_database_missing{int(100*p)}.csv"))
test_database = pd.read_csv(os.path.join(main_folder, f"test_database_missing{int(100*p)}.csv"), index_col=0)
for col in test_database.columns:
    try:
        test_database[col] = test_database[col].apply(eval)
    except: pass
print("test_database", test_database.shape)


########################################################
# Step 5: Training the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create the loaders.
train_data = RAGPTDataset(database=train_database)
train_dataloader = DataLoader(dataset= train_data, batch_size=batch_size,
                              collate_fn= RAGPTCollator(), shuffle=True)

test_data = RAGPTDataset(database=test_database)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                             collate_fn=RAGPTCollator(), shuffle=False)

########################################################
# Train the ``RAGPT`` model using the generated prompts. The model is trained for 10 epochs using PyTorch Lightning.
trainer = Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
estimator = RAGPT(cls_num=len(le.classes_))

# uncomment the line
# trainer.fit(estimator, train_dataloader, test_dataloader)

########################################################
# Step 6: Advanced Usage: Track Metrics During Training
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can modify the internal functions. For instance, we can track loss and compute evaluation metrics during
# training.

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

# uncomment the line
# trainer.fit(estimator, train_dataloader, test_dataloader)

########################################################
# After training, we can evaluate predictions.

# uncomment these lines
# preds = trainer.predict(estimator, DataLoader(dataset= train_data, batch_size=batch_size,
#                                               collate_fn= RAGPTCollator(), shuffle=False,
#                                               num_workers=8, pin_memory=True))
# preds = [batch.softmax(dim=1) for batch in preds]
# preds = [pred for batch in preds for pred in batch]
# preds = torch.stack(preds).argmax(1)
# print("Training metric:", matthews_corrcoef(y_true=y_train, y_pred=preds.cpu()))
#
# preds = trainer.predict(estimator, test_dataloader)
# preds = [batch.softmax(dim=1) for batch in preds]
# preds = [pred for batch in preds for pred in batch]
# preds = torch.stack(preds).argmax(1)
# print("Testing metric:", matthews_corrcoef(y_true=y_test, y_pred=preds.cpu()))

# results_dict = {
#     f"loss{int(100*p)}": [i.item() for i in estimator.agg_loss_list],
#     f"MCC{int(100*p)}": matthews_corrcoef(y_true=y_test, y_pred=preds.cpu()),
# }
# pd.Series(results_dict, name= int(100*p)).to_csv(os.path.join(main_folder, f"results{int(100*p)}.csv"))

########################################################
# After repeting the experiments with P=0, 0.3 and 0.7, we should have generated three result files that will use to
# visualize the learning curves.

results = pd.concat([pd.read_csv(os.path.join(main_folder, f"results{p}.csv"), index_col=0, names=[0], skiprows=1)
                     for p in [0, 30, 70]])
results_loss = results[results.index.str.startswith("loss")].squeeze().apply(eval).apply(pd.Series).T
results_metric = results[~results.index.str.startswith("loss")]
results_loss.columns = results_loss.columns.str.lstrip("loss") + "\% incomplete samples, MCC="
results_loss.columns = results_loss.columns + results_metric.squeeze().apply(eval).round(2).astype(str)
results_loss.columns = results_loss.columns.str.replace(r"^0\\% inc", "C", regex=True)
ax = results_loss.plot(xlabel="Epoch", ylabel="Loss (cross entropy)", figsize = (5, 3.5))
_ = ax.legend(handlelength=0.5, handletextpad=0.4)

###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We first built a memory bank with 5000 independent vision-language samples using the `iMML` `retrieve` module to
# generate retrieval-augmented prompts with a multi-channel retriever (``MCR``). Subsequently, we trained a model
# using the ``RAGPT`` algorithm available in `iMML` under three conditions: fully observed data, and scenarios with
# 30% and 70% randomly missing text and image modalities. The model demonstrated strong robustness on the test set,
# with only a slight drop in its performance with respect to the full data scenario, even under high levels of
# missing data.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This example illustrates how `iMML` enables state-of-the-art performance in classification, even in the presence
# of significant modality incompleteness in vision-language datasets.

