import itertools
import os.path
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from imvc.pipelines import NMFCPipeline, ConcatPipeline
from imvc.datasets import LoadDataset
from imvc.transformers.deepmf import DeepMFDataset, DeepMF
from imvc.utils import DatasetUtils
from imvc.transformers import ConcatenateViews
from imvc.utils.utils import BugInMONET
from utils.utils import save_record

random_state = 0
START_BENCHMARKING = True
folder_name = "results"
filelame = "complete_algorithms_evaluation.csv"
file_path = os.path.join(folder_name, filelame)

Xs, y = LoadDataset.load_incomplete_nutrimouse(p = 0, return_y = True, random_state = random_state)
nutrimouse_genotype = Xs, LabelEncoder().fit_transform(y[0]), y[0].squeeze().nunique(), "nutrimouse_genotype"
nutrimouse_diet = Xs, LabelEncoder().fit_transform(y[1]), y[1].squeeze().nunique(), "nutrimouse_diet"
datasets = [nutrimouse_genotype, nutrimouse_diet]
probs = np.arange(0., 1., step= 0.1).round(1)
algorithms = {
    "Concat": {"alg": ConcatPipeline, "params": {}},
    "NMFC": {"alg": NMFCPipeline, "params": {}},
}
runs_per_alg = np.arange(10).tolist()

iterations = itertools.product(algorithms.items(), datasets, probs, runs_per_alg)

if START_BENCHMARKING:
    results = pd.DataFrame()
else:
    results = pd.read_csv(file_path)

for (alg_name, alg_comp), (Xs, y, n_clusters, dataset_name), p, i in iterations:
    alg = alg_comp["alg"]
    params = alg_comp["params"]

    if not START_BENCHMARKING:
        checking_results = ((results["alg"] == alg_name) & (results["dataset"] == dataset_name)
                            & (results["missing_percentage"] == int(100*p)) & (results["execution"] == i))
        if not results[checking_results].empty:
            continue

    incomplete_Xs = DatasetUtils.convert_mvd_into_imvd(Xs=Xs, p=p, assess_percentage = True,
                                                       random_state = random_state + i)
    print(f"Algorithm: {alg_name} \t Dataset: {dataset_name} \t Missing: {p} \t Iteration: {i}")
    if alg_name == "MONET":
        model = alg(random_state = random_state + i, **params)
    elif alg_name in ["DeepMF"]:
        X = model[:-2].fit_transform(Xs)
        X = torch.from_numpy(X.values)
        train_data = DeepMFDataset(X=X)
        train_dataloader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
        trainer = Trainer(**params, logger=False, enable_checkpointing=False)
        ann = DeepMF(X=X)
        trainer.fit(ann, train_dataloader)
        incomplete_Xs = trainer.predict(ann, train_dataloader)
        model = model[-2:]

    else:
        model = alg(n_clusters=n_clusters, random_state = random_state + i, **params)
    model.estimator = model.estimator.set_params(verbose=False)
    errors_dict = defaultdict(int)
    while True:
        try:
            start_time = time.perf_counter()
            clusters = model.fit_predict(incomplete_Xs)
            break
        except BugInMONET as exception:
            model.estimator.set_params(random_state=model.estimator.random_state + 1)
            errors_dict[type(exception).__name__] += 1
            print(errors_dict)
            if sum(errors_dict.values()) == 100:
                break
        except NameError as exception:
            model.estimator.set_params(random_state=model.estimator.random_state + 1)
            errors_dict[type(exception).__name__] += 1
            print(errors_dict)
            if sum(errors_dict.values()) == 100:
                break
    if sum(errors_dict.values()) == 100:
        continue
    elapsed_time = time.perf_counter() - start_time
    if alg_name in ["MOFA", "Concat"]:
        X = make_pipeline(*model.transformers).transform(incomplete_Xs)
    elif alg_name in ["NMFC"]:
        X = model.transform(incomplete_Xs)
    elif alg_name in ["DeepMF"]:
        continue
    else:
        X = make_pipeline(*model.transformers).transform(incomplete_Xs)
        X = make_pipeline(ConcatenateViews(), SimpleImputer().set_output(transform="pandas")).fit_transform(X)

    mask = ~np.isnan(clusters)
    labels_pred = np.nan_to_num(clusters, nan = -1).astype(int)
    labels_pred = pd.factorize(labels_pred)[0]
    random_preds = [pd.Series(y).value_counts().index[0]] * len(y)

    dict_results = save_record(labels_pred=labels_pred, y=y, X=X, random_state=random_state, alg_name=alg_name,
                               dataset_name=dataset_name, p=p, elapsed_time=elapsed_time, i=i, random_preds=random_preds,
                               clusters=clusters, mask=mask, errors_dict=errors_dict)

    results = pd.concat([results, pd.DataFrame([dict_results])])
    results.to_csv(file_path, index=False)

