import os.path

import numpy as np
import torch
from pandarallel import pandarallel
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans

from imml.decomposition import MOFA, jNMF, DFMF
from imml.preprocessing import MultiViewTransformer, ConcatenateViews, NormalizerNaN
from imml.cluster import NEMO, DAIMC, PIMVC, SIMCADC, OSLFIMVC, MSNE, MKKMIK, LFIMVC, EEIMVC, SUMO, OPIMC, OMVC, MONET, \
    IMSR, IMSCAGL

from settings import FILLED_SUBRESULTS_PATH, FILLED_RESULTS_PATH, FILLED_LOGS_PATH, FILLED_ERRORS_PATH, \
    TIME_RESULTS_PATH, DATASET_TABLE_PATH, amputation_mechanisms, runs_per_alg, probs, \
    imputation, FILLED_RESULTS_FILE, FILLED_ERRORS_FILE, FILLED_SUBRESULTS_FOLDER, FILLED_LOGS_FILE
from src.commons import CommonOperations

args = CommonOperations.get_args()

if not args.save_results:
    results_folder = 'test'
    FILLED_RESULTS_PATH = os.path.join(results_folder, FILLED_RESULTS_FILE)
    FILLED_LOGS_PATH = os.path.join(results_folder, FILLED_LOGS_FILE)
    FILLED_ERRORS_PATH = os.path.join(results_folder, FILLED_ERRORS_FILE)
    FILLED_SUBRESULTS_PATH = os.path.join(results_folder, FILLED_SUBRESULTS_FOLDER)

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

algorithms = {
    "DAIMC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   DAIMC()), "params": {}},
    "EEIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    EEIMVC()), "params": {}},
    "NEMO": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    NEMO()), "params": {}},
    # "IMSCAGL": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
    #                               MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
    #                                IMSCAGL()), "params": {}},
    "IMSR": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   IMSR()), "params": {}},
    "LFIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    LFIMVC()), "params": {}},
    "MKKMIK": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    MKKMIK()), "params": {}},
    "MRGCN": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                   MultiViewTransformer(SimpleImputer(strategy= "constant",
                                                                      fill_value=0.0).set_output(transform="pandas")),
                                   MultiViewTransformer(FunctionTransformer(
                                       lambda x: torch.from_numpy(x.values.astype(np.float32))))),
              "params": {}},
    # "MONET": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
    #                                MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                                MONET()), "params": {}},
    # "MSNE": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
    #                               MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                               MSNE()), "params": {}},
    "OMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  OMVC()), "params": {}},
    "OPIMC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   OPIMC()), "params": {}},
    "OSLFIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                      MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                      OSLFIMVC()), "params": {}},
    "PIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   PIMVC()), "params": {}},
    "SIMCADC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                     MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                     SIMCADC()), "params": {}},
    # "SUMO": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
    #                               MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                               SUMO()), "params": {}},
    # "DeepMF": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
    #                                 ConcatenateViews(), StandardScaler(),
    #                                 FunctionTransformer(lambda x: torch.from_numpy(x).float().cuda().t()),
    #                                 DeepMF(), FunctionTransformer(lambda x: x.cpu().detach().numpy()),
    #                                 StandardScaler(), KMeans(n_init= "auto")),
    #              "params": {}},
    "DFMF": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  DFMF().set_output(transform="pandas"),
                                  StandardScaler().set_output(transform="pandas"), KMeans(n_init= "auto")),
             "params": {}},
    "MOFA": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  MOFA().set_output(transform="pandas"),
                                  StandardScaler().set_output(transform="pandas"), KMeans(n_init= "auto")),
             "params": {}},
    "jNMF": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")),
                                  jNMF().set_output(transform="pandas"),
                                  StandardScaler().set_output(transform="pandas"), KMeans(n_init= "auto")),
             "params": {}},
}
incomplete_algorithms = False
CommonOperations.run_script(dataset_table_path=DATASET_TABLE_PATH, algorithms=algorithms, probs=probs,
                            amputation_mechanisms=amputation_mechanisms, imputation=imputation,
                            runs_per_alg=runs_per_alg, args=args, subresults_path=FILLED_SUBRESULTS_PATH,
                            logs_file=FILLED_LOGS_PATH, error_file=FILLED_ERRORS_PATH,
                            results_path=FILLED_RESULTS_PATH, time_results_path=TIME_RESULTS_PATH,
                            incomplete_algorithms=incomplete_algorithms)

