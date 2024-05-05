import itertools
import json
import os.path
from collections import defaultdict
import numpy as np
from imvc.datasets import LoadDataset
from imvc.transformers import Amputer
from imvc.utils import DatasetUtils
from settings import PROFILES_PATH

if os.path.exists(PROFILES_PATH):
    os.remove(PROFILES_PATH)

random_state = 42

datasets = [
    "simulated_gm",
    "simulated_InterSIM",
    "simulated_netMUG",
    "nutrimouse_genotype",
    "nutrimouse_diet",
    "bbcsport",
    "buaa",
    "metabric",
    "digits",
    "bdgp",
    "tcga",
    "caltech101",
    "nuswide",
]
two_view_datasets = ["simulated_gm", "nutrimouse_genotype", "nutrimouse_diet", "metabric", "bdgp",
                     "buaa", "simulated_netMUG"]
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
    Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle= False)
    y = y[y_name]
    n_clusters = y.nunique()

    for prob, amputation_mechanism, run_n in itertools.product(probs, amputation_mechanisms, runs_per_alg):
        if (dataset_name in two_view_datasets) and (amputation_mechanism in ["MAR", "MNAR"]):
            continue
        train_Xs = DatasetUtils.shuffle_imvd(Xs=Xs, random_state=random_state + run_n)
        y_train = y.loc[train_Xs[0].index]
        strat = False
        p = prob/100

        if p != 0:
            if amputation_mechanism == "EDM":
                try:
                    assert n_clusters < len(train_Xs[0]) * (1 - p)
                except AssertionError as exception:
                    raise AssertionError(f"{exception}; n_clusters < len(train_Xs[0]) * (1-p)")
                amp = Amputer(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state + run_n,
                              assess_percentage=True, stratify=y_train)
                try:
                    train_Xs = amp.fit_transform(train_Xs)
                    strat = True
                except ValueError:
                    amp.set_params(**{"stratify": None})
                    train_Xs = amp.fit_transform(train_Xs)
            else:
                amp = Amputer(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state + run_n)
                train_Xs = amp.fit_transform(train_Xs)
        else:
            amputation_mechanism = "'None'"

        dict_indxs[dataset_name][int(prob)][amputation_mechanism][int(run_n)] = {
            "stratify": strat,
            "missing_view_profile": DatasetUtils.get_missing_view_profile(train_Xs).to_dict(),
        }

        with open(PROFILES_PATH, 'w') as fp:
            json.dump(dict_indxs, fp)

print("Completed successfully!")


