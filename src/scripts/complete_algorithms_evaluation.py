import os.path
import dill
from pandarallel import pandarallel
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans
from mvlearn.decomposition import AJIVE, GroupPCA
from mvlearn.cluster import MultiviewSpectralClustering, MultiviewCoRegSpectralClustering
from imvc.preprocessing import MultiViewTransformer, ConcatenateViews
from imvc.algorithms import NMFC
from settings import SUBRESULTS_PATH, COMPLETE_RESULTS_PATH, COMPLETE_LOGS_PATH, COMPLETE_ERRORS_PATH, \
    TIME_RESULTS_PATH, DATASET_TABLE_PATH, PROFILES_PATH, amputation_mechanisms, runs_per_alg, probs, \
    imputation
from src.utils import CommonOperations

args = CommonOperations.get_args()

if not args.save_results:
    COMPLETE_RESULTS_PATH = os.path.join("test", "incomplete_algorithms_evaluation.csv")
    COMPLETE_LOGS_PATH = os.path.join("test", "incomplete_logs.txt")
    COMPLETE_ERRORS_PATH = os.path.join("test", "incomplete_errors.txt")
    SUBRESULTS_PATH = os.path.join("test", "subresults")

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

with open(PROFILES_PATH, 'rb') as f:
    profile_missing = dill.load(f)

algorithms = {
    "Concat": {"alg": make_pipeline(ConcatenateViews(),
                                    StandardScaler().set_output(transform='pandas'),
                                    KMeans()), "params": {}},
    "NMFC": {"alg": make_pipeline(ConcatenateViews(),
                                  MinMaxScaler().set_output(transform='pandas'),
                                  NMFC().set_output(transform='pandas')), "params": {}},
    "MVSpectralClustering": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                                  MultiviewSpectralClustering()),
                             "params": {}},
    "MVCoRegSpectralClustering": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform= "pandas")),
                                                       MultiviewCoRegSpectralClustering()),
                                  "params": {}},
    "GroupPCA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), GroupPCA(), StandardScaler(), KMeans()),
                 "params": {}},
    "AJIVE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler()), AJIVE(),
                                   MultiViewTransformer(FunctionTransformer(pd.DataFrame)), ConcatenateViews(),
                                   StandardScaler(), KMeans()),
              "params": {}},
    "SNF": {"alg": MultiViewTransformer(StandardScaler().set_output(transform="pandas")), "params": {}},
}
incomplete_algorithms = False
CommonOperations.run_script(dataset_table_path=DATASET_TABLE_PATH, algorithms=algorithms, probs=probs,
                            amputation_mechanisms=amputation_mechanisms, imputation=imputation,
                            runs_per_alg=runs_per_alg, args=args, subresults_path=SUBRESULTS_PATH,
                            logs_file=COMPLETE_LOGS_PATH, error_file=COMPLETE_ERRORS_PATH,
                            results_path=COMPLETE_RESULTS_PATH, time_results_path=TIME_RESULTS_PATH,
                            incomplete_algorithms=incomplete_algorithms, profile_missing=profile_missing)

