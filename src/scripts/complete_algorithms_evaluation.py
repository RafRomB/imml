import os.path
from pandarallel import pandarallel

from settings import COMPLETE_SUBRESULTS_PATH, COMPLETE_RESULTS_PATH, COMPLETE_LOGS_PATH, COMPLETE_ERRORS_PATH, \
    TIME_RESULTS_PATH, DATASET_TABLE_PATH, amputation_mechanisms, runs_per_alg, probs, \
    imputation, COMPLETE_RESULTS_FILE, COMPLETE_ERRORS_FILE, COMPLETE_SUBRESULTS_FOLDER, COMPLETE_LOGS_FILE
from src.commons import CommonOperations
from src.models import complete_algorithms, incomplete_algorithms

algorithms = {**complete_algorithms, **incomplete_algorithms}

args = CommonOperations.get_args()

if not args.save_results:
    results_folder = 'test'
    COMPLETE_RESULTS_PATH = os.path.join(results_folder, COMPLETE_RESULTS_FILE)
    COMPLETE_LOGS_PATH = os.path.join(results_folder, COMPLETE_LOGS_FILE)
    COMPLETE_ERRORS_PATH = os.path.join(results_folder, COMPLETE_ERRORS_FILE)
    COMPLETE_SUBRESULTS_PATH = os.path.join(results_folder, COMPLETE_SUBRESULTS_FOLDER)

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

incomplete_algorithms = False
CommonOperations.run_script(dataset_table_path=DATASET_TABLE_PATH, algorithms=algorithms, probs=probs,
                            amputation_mechanisms=amputation_mechanisms, imputation=imputation,
                            runs_per_alg=runs_per_alg, args=args, subresults_path=COMPLETE_SUBRESULTS_PATH,
                            logs_file=COMPLETE_LOGS_PATH, error_file=COMPLETE_ERRORS_PATH,
                            results_path=COMPLETE_RESULTS_PATH, time_results_path=TIME_RESULTS_PATH,
                            incomplete_algorithms=incomplete_algorithms)

