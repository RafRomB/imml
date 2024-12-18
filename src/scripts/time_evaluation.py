import argparse
import os
import time
from datetime import datetime
import pandas as pd

from src.models import Model
from settings import TIME_RESULTS_PATH, TIME_LOGS_PATH, TIME_ERRORS_PATH, RANDOM_STATE, DATASET_TABLE_PATH
from src.commons import CommonOperations
from src.models import incomplete_algorithms, complete_algorithms


datasets = pd.read_csv(DATASET_TABLE_PATH)["dataset"].to_list()
if "nutrimouse" in datasets:
    pos = datasets.index('nutrimouse')
    datasets[pos:pos + 1] = ('nutrimouse_genotype', 'nutrimouse_diet')

algorithms = {**complete_algorithms, **incomplete_algorithms}

# args = lambda: None
# args.continue_benchmarking, args.n_jobs, args.save_results = True, 2, False
parser = argparse.ArgumentParser()
parser.add_argument('-save_results', default= False, action='store_true')
args = parser.parse_args()
if not args.save_results:
    TIME_RESULTS_PATH = os.path.join("test", "time_evaluation.csv")
    TIME_LOGS_PATH = os.path.join("test", "time_logs.txt")
    TIME_ERRORS_PATH = os.path.join("test", "time_errors.txt")

if os.path.exists(TIME_RESULTS_PATH):
    results = pd.read_csv(TIME_RESULTS_PATH)
else:
    results = pd.DataFrame(algorithms.keys(), columns=["algorithm"]).merge(pd.DataFrame(datasets, columns=["dataset"]),
                                                                           how='cross')
    results["time"] = -1
    results["finished"] = False
    results["completed"] = False
    results["comments"] = "{}"

    os.remove(TIME_LOGS_PATH) if os.path.exists(TIME_LOGS_PATH) else None
    os.remove(TIME_ERRORS_PATH) if os.path.exists(TIME_ERRORS_PATH) else None
    open(TIME_LOGS_PATH, 'w').close()
    open(TIME_ERRORS_PATH, 'w').close()

for dataset_name in datasets:
    Xs, y, n_clusters = CommonOperations.load_Xs_y(dataset_name=dataset_name)

    for idx, row in results[(~results["finished"]) & (results["dataset"] == dataset_name)].iterrows():
        alg_name = row["algorithm"]
        if alg_name in ["IntNMF", "COCA", "JNMF"]:
            continue
        alg = algorithms[alg_name]
        with open(TIME_LOGS_PATH, "a") as f:
            f.write(f'\n {dataset_name} \t {alg_name} \t {datetime.now()}')

        start_time = time.perf_counter()
        try:
            clusters, _ = Model(alg_name=alg_name, alg=alg).method(train_Xs=Xs, n_clusters=n_clusters,
                                                                   random_state=RANDOM_STATE, run_n=0)
            elapsed_time = time.perf_counter() - start_time
            errors_dict = {}
            completed = True
        except Exception as exception:
            elapsed_time = time.perf_counter() - start_time
            exception_name = type(exception).__name__
            errors_dict = {f"{exception_name}": exception}
            with open(TIME_ERRORS_PATH, "a") as f:
                f.write(f'\n {dataset_name} \t {alg_name} \t {exception_name}: {exception} \t {datetime.now()}')
            completed = False

        results.loc[idx, ["time", "finished", "completed", "comments"]] = [elapsed_time, True, completed, errors_dict]
        results.to_csv(TIME_RESULTS_PATH, index=False)

print("Completed successfully!")
with open(TIME_LOGS_PATH, "a") as f:
    f.write(f'\n Completed successfully \t {datetime.now()}')

