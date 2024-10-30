import os.path
from pandarallel import pandarallel

from settings import FILLED_SUBRESULTS_PATH, FILLED_RESULTS_PATH, FILLED_LOGS_PATH, FILLED_ERRORS_PATH, \
    TIME_RESULTS_PATH, DATASET_TABLE_PATH, amputation_mechanisms, runs_per_alg, probs, \
    imputation, FILLED_RESULTS_FILE, FILLED_ERRORS_FILE, FILLED_SUBRESULTS_FOLDER, FILLED_LOGS_FILE
from src.commons import CommonOperations
from src.scripts.incomplete_algorithms_evaluation import algorithms

args = CommonOperations.get_args()

if not args.save_results:
    results_folder = 'test'
    FILLED_RESULTS_PATH = os.path.join(results_folder, FILLED_RESULTS_FILE)
    FILLED_LOGS_PATH = os.path.join(results_folder, FILLED_LOGS_FILE)
    FILLED_ERRORS_PATH = os.path.join(results_folder, FILLED_ERRORS_FILE)
    FILLED_SUBRESULTS_PATH = os.path.join(results_folder, FILLED_SUBRESULTS_FOLDER)

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

incomplete_algorithms = False
CommonOperations.run_script(dataset_table_path=DATASET_TABLE_PATH, algorithms=algorithms, probs=probs,
                            amputation_mechanisms=amputation_mechanisms, imputation=imputation,
                            runs_per_alg=runs_per_alg, args=args, subresults_path=FILLED_SUBRESULTS_PATH,
                            logs_file=FILLED_LOGS_PATH, error_file=FILLED_ERRORS_PATH,
                            results_path=FILLED_RESULTS_PATH, time_results_path=TIME_RESULTS_PATH,
                            incomplete_algorithms=incomplete_algorithms)

