import os.path

import numpy as np
import torch
from pandarallel import pandarallel
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from imml.decomposition import DFMF, MOFA, DeepMF, jNMF
from imml.preprocessing import MultiViewTransformer, ConcatenateViews, NormalizerNaN
from imml.cluster import NEMO, DAIMC, PIMVC, SIMCADC, OSLFIMVC, MSNE, MKKMIK, LFIMVC, EEIMVC, SUMO, OPIMC, OMVC, MONET, \
    IMSR, IMSCAGL

from settings import INCOMPLETE_RESULTS_PATH, INCOMPLETE_SUBRESULTS_PATH, INCOMPLETE_LOGS_PATH, INCOMPLETE_ERRORS_PATH, \
    TIME_RESULTS_PATH, DATASET_TABLE_PATH, amputation_mechanisms, probs, \
    imputation, runs_per_alg, INCOMPLETE_RESULTS_FILE, INCOMPLETE_LOGS_FILE, INCOMPLETE_ERRORS_FILE, \
    INCOMPLETE_SUBRESULTS_FOLDER
from src.commons import CommonOperations

args = CommonOperations.get_args()

if not args.save_results:
    results_folder = 'test'
    INCOMPLETE_RESULTS_PATH = os.path.join(results_folder, INCOMPLETE_RESULTS_FILE)
    INCOMPLETE_LOGS_PATH = os.path.join(results_folder, INCOMPLETE_LOGS_FILE)
    INCOMPLETE_ERRORS_PATH = os.path.join(results_folder, INCOMPLETE_ERRORS_FILE)
    INCOMPLETE_SUBRESULTS_PATH = os.path.join(results_folder, INCOMPLETE_SUBRESULTS_FOLDER)

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

algorithms = {
    "DAIMC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   DAIMC()), "language": "Python"},
    "EEIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    EEIMVC()), "language": "Python"},
    "NEMO": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    NEMO()), "language": "Python"},
    "IMSCAGL": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   IMSCAGL()), "language": "Matlab"},
    "IMSR": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   IMSR()), "language": "Python"},
    "LFIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    LFIMVC()), "language": "Python"},
    "MKKMIK": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    MKKMIK()), "language": "Matlab"},
    "MRGCN": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                   MultiViewTransformer(SimpleImputer(strategy= "constant",
                                                                      fill_value=0.0).set_output(transform="pandas")),
                                   MultiViewTransformer(FunctionTransformer(
                                       lambda x: torch.from_numpy(x.values.astype(np.float32))))),
              "language": "DL"},
    "MONET": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                   MONET()), "language": "Python"},
    "MSNE": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  MSNE()), "language": "Python"},
    "OMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  OMVC()), "language": "Matlab"},
    "OPIMC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   OPIMC()), "language": "Matlab"},
    "OSLFIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                      MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                      OSLFIMVC()), "language": "Matlab"},
    "PIMVC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                   MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   PIMVC()), "language": "Matlab"},
    "SIMCADC": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                     MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                     SIMCADC()), "language": "Python"},
    "SUMO": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  SUMO()), "language": "Python"},
    "DeepMF": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                    ConcatenateViews(), StandardScaler(),
                                    FunctionTransformer(lambda x: torch.from_numpy(x).float().cuda().t()),
                                    DeepMF, FunctionTransformer(lambda x: x.cpu().detach().numpy()),
                                    StandardScaler(), KMeans(n_init= "auto")),
                 "language": "DL"},
    "DFMF": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  DFMF(),
                                  StandardScaler().set_output(transform="pandas"), KMeans(n_init= "auto")),
             "language": "Python"},
    "MOFA": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  MOFA(),
                                  StandardScaler().set_output(transform="pandas"), KMeans(n_init= "auto")),
             "language": "Python"},
    "jNMF": {"alg": make_pipeline(MultiViewTransformer(VarianceThreshold().set_output(transform="pandas")),
                                  MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")),
                                  jNMF(),
                                  StandardScaler().set_output(transform="pandas"), KMeans(n_init= "auto")),
             "language": "R"},
}
incomplete_algorithms = True
CommonOperations.run_script(dataset_table_path=DATASET_TABLE_PATH, algorithms=algorithms, probs=probs,
                            amputation_mechanisms=amputation_mechanisms, imputation=imputation,
                            runs_per_alg=runs_per_alg, args=args, subresults_path=INCOMPLETE_SUBRESULTS_PATH,
                            logs_file=INCOMPLETE_LOGS_PATH, error_file=INCOMPLETE_ERRORS_PATH,
                            results_path=INCOMPLETE_RESULTS_PATH, time_results_path=TIME_RESULTS_PATH,
                            incomplete_algorithms=incomplete_algorithms)
