import itertools
import os.path
from collections import defaultdict
import dill
import numpy as np
from sklearn.utils import shuffle
from imvc.ampute import Amputer
from imvc.impute import get_observed_view_indicator

from settings import PROFILES_PATH, DATASET_TABLE_PATH, RANDOM_STATE, probs, amputation_mechanisms, runs_per_alg
from src.utils import CommonOperations

if os.path.exists(PROFILES_PATH):
    os.remove(PROFILES_PATH)

datasets, two_view_datasets = CommonOperations.get_datasets(DATASET_TABLE_PATH)
infinite_defaultdict = lambda: defaultdict(infinite_defaultdict)
dict_indxs = infinite_defaultdict()


for dataset_name in datasets:
    Xs, y, n_clusters = CommonOperations.get_dataset_by_name(dataset_name=dataset_name)

    for prob, amputation_mechanism, run_n in itertools.product(probs, amputation_mechanisms, runs_per_alg):
        print(dataset_name, prob, amputation_mechanism, run_n)
        try:
            random_state = RANDOM_STATE + run_n
            if (dataset_name in two_view_datasets) and (amputation_mechanism in ["MAR", "MNAR"]):
                continue
            *train_Xs, y_train = shuffle(*Xs, y, random_state=random_state)
            p = prob/100

            if p != 0:
                amp = Amputer(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state)
                train_Xs = amp.fit_transform(train_Xs)
            else:
                amputation_mechanism = "No"

            observed_view_indicator = get_observed_view_indicator(train_Xs)
            observed_view_indicator.index = observed_view_indicator.index.astype(np.int8)
            dict_indxs[dataset_name][int(prob)][amputation_mechanism][int(run_n)] = {
                "observed_view_indicator": observed_view_indicator.to_dict(),
                "valid": True,
            }

        except ValueError as exception:
            dict_indxs[dataset_name][int(prob)][amputation_mechanism][int(run_n)] = {
                "valid" : False,
                "error": str(exception),
            }

with open(PROFILES_PATH, 'wb') as fp:
    dill.dump(dict_indxs, fp, dill.HIGHEST_PROTOCOL)

print("Completed successfully!")


