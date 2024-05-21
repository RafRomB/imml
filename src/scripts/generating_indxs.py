import itertools
import json
import os.path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from imvc.datasets import LoadDataset
from imvc.ampute import Amputer
from imvc.impute import get_observed_view_indicator

from settings import PROFILES_PATH, DATASET_TABLE_PATH, RANDOM_STATE

if os.path.exists(PROFILES_PATH):
    os.remove(PROFILES_PATH)

dataset_table = pd.read_csv(DATASET_TABLE_PATH)
dataset_table = dataset_table.reindex(dataset_table.index.append(dataset_table.index[dataset_table["dataset"]=="nutrimouse"])).sort_index().reset_index(drop=True)
dataset_table.loc[dataset_table["dataset"] == "nutrimouse", "dataset"] = ["nutrimouse_genotype", "nutrimouse_diet"]
datasets = dataset_table["dataset"].to_list()
two_view_datasets = dataset_table[dataset_table["n_features"].apply(lambda x: len(x) == 2)]
amputation_mechanisms = ["EDM", 'MCAR', 'MAR', 'MNAR', "PM"]
probs = np.arange(100, step= 10)
runs_per_alg = np.arange(50)

infinite_defaultdict = lambda: defaultdict(infinite_defaultdict)
dict_indxs = infinite_defaultdict()


for dataset_name in datasets:
    names = dataset_name.split("_")
    if "simulated" in names:
        names = ["_".join(names)]
    x_name,y_name = names if len(names) > 1 else (names[0], "0")
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True)
    y = y[y_name]
    n_clusters = y.nunique()

    for prob, amputation_mechanism, run_n in itertools.product(probs, amputation_mechanisms, runs_per_alg):
        random_state = RANDOM_STATE + run_n
        if (dataset_name in two_view_datasets) and (amputation_mechanism in ["MAR", "MNAR"]):
            continue
        *train_Xs, y_train = shuffle(*Xs, y, random_state=random_state)
        strat = False
        p = prob/100

        if p != 0:
            if amputation_mechanism == "EDM":
                try:
                    assert n_clusters < len(train_Xs[0]) * (1 - p)
                except AssertionError as exception:
                    raise AssertionError(f"{exception}; n_clusters < len(train_Xs[0]) * (1-p)")
                amp = Amputer(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state,
                              assess_percentage=True, stratify=y_train)
                try:
                    train_Xs = amp.fit_transform(train_Xs)
                    strat = True
                except ValueError:
                    amp.set_params(**{"stratify": None})
                    train_Xs = amp.fit_transform(train_Xs)
            else:
                amp = Amputer(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state)
                train_Xs = amp.fit_transform(train_Xs)
        else:
            amputation_mechanism = "No"

        dict_indxs[dataset_name][int(prob)][amputation_mechanism][int(run_n)] = {
            "stratify": strat,
            "observed_view_indicator": get_observed_view_indicator(train_Xs).to_dict(),
        }

        with open(PROFILES_PATH, 'w') as fp:
            json.dump(dict_indxs, fp)

print("Completed successfully!")


